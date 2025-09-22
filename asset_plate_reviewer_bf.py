import os
import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from threading import Lock
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify

# --- Resolve paths relative to this repository for templates/static ---
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "review_asset_templates"

# Prefer "review_asset_templates/static"; fall back to repo-level "static"; else disable Flask static
CANDIDATE_STATIC = [
    TEMPLATE_DIR / "static",
    BASE_DIR / "static",
]
STATIC_DIR = next((p for p in CANDIDATE_STATIC if p.exists()), None)

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR) if STATIC_DIR else None,  # None if served by Nginx
)

# --- Paths ---
JSON_DIR = r"/home/developer/Output_jason_api"  # Linux path for Docker deployment
IMG_DIR  = r"/home/developer/Capture_photos_upload" # Linux path for Docker deployment

# --- SQLite DB (for dropdown options & syncs) ---
DB_PATH = r"/home/developer/asset_capture_app_dev/data/QR_codes.db"

# Tables/columns
ASSET_GROUP_TABLE = "Asset_Group"
ASSET_GROUP_COL   = "name"

# --- Application table autodetection ---
APPLICATION_TABLE_CANDIDATES = [
    "bf_applicaton_type",   # original
    "bf_application_type",  # fixed spelling
    "BF_Application_Type"   # possible title-case
]
TEXT_TYPE_HINTS = {"TEXT", "VARCHAR", "NVARCHAR", "CHAR", "CLOB"}

VALID_IMAGE_EXTS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

# Required-photo sequences (TSBC -3 excluded everywhere)
SEQ_CHECK = ['-0', '-1', '-2']  # Asset Plate/Label, Asset Plate/Label (Optional), Main Asset
SEQ_SHOW  = ['-0', '-1', '-2']

# JSON filename pattern: "<QR>_<TYPE>_<Building>.json"
JSON_NAME_RE = re.compile(r"^(\d+)_([A-Za-z]+)_(\d+(?:-\d+)?)\.json$")


# --- START: Directory Sync Logic ---

# --- Image Sync ---
DATA_DIR = Path(DB_PATH).parent
PROCESSED_LOG_BF = DATA_DIR / "processed_images_bf.log"
IMG_NAME_RE_BF = re.compile(r"^(\d+)\s+(.+?)\s+BF\s+-\s+[0-3]\.(?:jpe?g|png)$", re.IGNORECASE)
image_sync_lock = Lock()

# --- JSON Sync ---
PROCESSED_JSON_LOG_BF = DATA_DIR / "processed_json_bf.log"
json_sync_lock = Lock()


def _is_bf_filename(filename: str) -> bool:
    """True if the JSON is for a BF asset type."""
    if not filename.endswith(".json"): return False
    m = JSON_NAME_RE.match(filename)
    if not m:
        return False
    _qr, asset_type_mid, _building = m.groups()
    return asset_type_mid.upper() == "BF"


def sync_image_directory_to_db_bf():
    """
    Scans IMG_DIR for new BF image files and upserts placeholder entries into sdi_dataset.
    """
    if not image_sync_lock.acquire(blocking=False):
        return

    try:
        if not os.path.isdir(IMG_DIR):
            return

        processed_files = set()
        if PROCESSED_LOG_BF.exists():
            with open(PROCESSED_LOG_BF, 'r', encoding='utf-8') as f:
                processed_files = {line.strip() for line in f if line.strip()}

        current_files = {f for f in os.listdir(IMG_DIR) if f.lower().endswith(tuple(VALID_IMAGE_EXTS))}
        new_files = sorted(list(current_files - processed_files))

        if not new_files:
            return

        print(f"SYNC-IMG (BF): Found {len(new_files)} new image(s).")
        successfully_processed = []
        for filename in new_files:
            match = IMG_NAME_RE_BF.match(filename)
            if not match:
                # Log non-matching files so they are not re-checked
                if " BF " in filename:
                    successfully_processed.append(filename)
                continue

            qr, building = match.groups()
            try:
                # Construct the doc_id that the existing upsert function expects
                doc_id = f"{qr.strip()}_BF_{building.strip()}"
                # Call the existing upsert function with an empty dict to create a placeholder
                upsert_sdi_dataset(doc_id=doc_id, structured={})
                successfully_processed.append(filename)
            except Exception as e:
                print(f"SYNC-IMG-ERROR (BF): DB upsert failed for {filename}: {e}")

        if successfully_processed:
            with open(PROCESSED_LOG_BF, 'a', encoding='utf-8') as f:
                for filename in successfully_processed:
                    f.write(f"{filename}\n")
    finally:
        image_sync_lock.release()


def sync_json_directory_to_db_bf():
    """
    Scans JSON_DIR for new or modified BF JSON files and upserts their structured data.
    """
    if not json_sync_lock.acquire(blocking=False):
        return

    try:
        if not os.path.isdir(JSON_DIR):
            return

        processed_files = {}
        if PROCESSED_JSON_LOG_BF.exists():
            with open(PROCESSED_JSON_LOG_BF, 'r', encoding='utf-8') as f:
                try:
                    processed_files = json.load(f)
                except json.JSONDecodeError:
                    print("SYNC-JSON-WARN (BF): Could not read log, starting fresh.")

        files_to_process = {}
        for filename in os.listdir(JSON_DIR):
            if not _is_bf_filename(filename):
                continue
            
            filepath = os.path.join(JSON_DIR, filename)
            current_mtime = os.path.getmtime(filepath)
            
            if filename not in processed_files or current_mtime > processed_files[filename]:
                files_to_process[filename] = current_mtime

        if not files_to_process:
            return

        print(f"SYNC-JSON (BF): Found {len(files_to_process)} new/updated JSON file(s).")
        for filename, mtime in files_to_process.items():
            doc_id = filename[:-5]
            try:
                with open(os.path.join(JSON_DIR, filename), 'r', encoding='utf-8') as f:
                    content = json.load(f)
                
                structured_data = content.get("structured_data", {})
                if isinstance(structured_data, dict):
                    print(f"   -> Syncing data from {filename}")
                    upsert_sdi_dataset(doc_id=doc_id, structured=structured_data)
                    processed_files[filename] = mtime
                else:
                    processed_files[filename] = mtime

            except Exception as e:
                print(f"SYNC-JSON-ERROR (BF): Failed to process {filename}: {e}")
        
        with open(PROCESSED_JSON_LOG_BF, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, indent=2)

    finally:
        json_sync_lock.release()


@app.before_request
def before_request_handler():
    """
    Runs sync logic before each request for BF assets.
    """
    if request.endpoint in ('static', 'serve_image', 'check_sdi'):
        return
    sync_image_directory_to_db_bf()
    sync_json_directory_to_db_bf()

# --- END: Directory Sync Logic ---


def find_image(qr: str, building: str, asset_type_mid: str, seq_tag: str):
    seq = seq_tag.replace('-', '').strip()
    base = f"{qr} {building} {asset_type_mid} - {seq}"
    for ext in VALID_IMAGE_EXTS:
        candidate = os.path.join(IMG_DIR, base + ext)
        if os.path.exists(candidate):
            return os.path.basename(candidate)
    return None


@lru_cache(maxsize=1)
def _connectable():
    return os.path.exists(DB_PATH)


def _fetch_column_values(table: str, col: str):
    if not _connectable():
        return []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            query = f'SELECT "{col}" AS val FROM "{table}" WHERE "{col}" IS NOT NULL'
            cur.execute(query)
            vals = [str(r["val"]).strip() for r in cur.fetchall() if str(r["val"]).strip()]
            uniq = sorted(set(vals), key=lambda s: (s.lower(), s))
            return uniq
    except Exception as e:
        print(f"⚠️ DB fetch failed for {table}.{col}: {e}")
        return []


def get_asset_group_options():
    all_opts = _fetch_column_values(ASSET_GROUP_TABLE, ASSET_GROUP_COL)
    allowed = {"Primary Backflow Devices", "Secondary Backflow Devices"}
    return [opt for opt in all_opts if opt in allowed]


def get_attribute_options():
    return ["BackflowDevice"]


def _table_exists(conn, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    return cur.fetchone() is not None


def _text_columns(conn, table_name: str):
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table_name}")')
    cols = cur.fetchall()
    if not cols:
        return []
    text_cols, all_cols = [], []
    for cid, name, coltype, notnull, dflt, pk in cols:
        all_cols.append(name)
        t = (coltype or "").upper()
        if any(hint in t for hint in TEXT_TYPE_HINTS) or t == "":
            text_cols.append(name)
        return text_cols or all_cols


def _fetch_distinct_nonempty(conn, table: str, col: str):
    cur = conn.cursor()
    try:
        cur.execute(f'SELECT DISTINCT "{col}" FROM "{table}"')
        vals = []
        for (v,) in cur.fetchall():
            if v is None:
                continue
            s = str(v).strip()
            if s:
                vals.append(s)
        vals = sorted(set(vals), key=lambda s: (s.lower(), s))
        return vals
    except Exception as e:
        print(f"⚠️ Failed fetching distinct from {table}.{col}: {e}")
        return []


def get_application_options():
    if not _connectable():
        print("⚠️ DB not found:", DB_PATH)
        return []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            for tbl in APPLICATION_TABLE_CANDIDATES:
                if not _table_exists(conn, tbl):
                    continue
                candidate_cols = ["name", "Name", "code", "Code", "application", "Application", "type", "Type"]
                cols = _text_columns(conn, tbl)
                ordered_cols = [c for c in candidate_cols if c in cols] + [c for c in cols if c not in candidate_cols]
                for col in ordered_cols:
                    vals = _fetch_distinct_nonempty(conn, tbl, col)
                    if vals:
                        print(f"✔️ Application options from {tbl}.{col}: {len(vals)} values")
                        return vals
                print(f"ℹ️ Table '{tbl}' found, but no non-empty text-like columns produced values: {cols}")
            print("⚠️ No Application table/values found. Tried:", APPLICATION_TABLE_CANDIDATES)
            return []
    except Exception as e:
        print("❌ Error while loading Application options:", e)
        return []


# ===== Description logic =====
def _compute_description(asset_group: str, ubc_tag: str, application: str = "") -> str:
    prefix = "BFP-"
    suffix = "BFP"
    app_part = (application or "").strip()
    if app_part:
        app_part += " "
    first_word_group = ""
    if asset_group:
        first_word_group = asset_group.strip().split()[0] + " "
    return f"{prefix}{app_part}{first_word_group}{suffix}".strip()


def parse_doc_id(doc_id: str):
    m = JSON_NAME_RE.match(f"{doc_id}.json")
    if not m:
        return None
    return m.groups()


# ====== DB helpers: QR_codes.Approved and SDI dataset sync ======
def upsert_approved_in_db(qr_code: str, is_approved: bool):
    if not _connectable():
        return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute('CREATE TABLE IF NOT EXISTS "QR_codes" ("QR_code_ID" TEXT PRIMARY KEY, "Approved" TEXT)')
            new_val = '1' if is_approved else ''
            cur.execute(
                'INSERT INTO "QR_codes" ("QR_code_ID","Approved") VALUES (?,?) '
                'ON CONFLICT("QR_code_ID") DO UPDATE SET "Approved"=excluded."Approved"',
                (qr_code, new_val)
            )
            conn.commit()
    except Exception as e:
        print(f"⚠️ Failed to update QR_codes.Approved for {qr_code}: {e}")


def _table_columns(conn, table_name: str):
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table_name}")')
    return {row[1] for row in cur.fetchall()}


def _safe_str(v):
    return "" if v is None else str(v)


def _build_sdi_row(doc_id: str, structured: dict) -> dict:
    parsed = parse_doc_id(doc_id)
    if not parsed:
        return {}
    qr, _asset_type_mid, building = parsed

    manuf = _safe_str(structured.get("Manufacturer", ""))
    model = _safe_str(structured.get("Model", ""))
    serial = _safe_str(structured.get("Serial Number", ""))
    ubc = _safe_str(structured.get("UBC Tag", ""))
    ag = _safe_str(structured.get("Asset Group", ""))
    attr = _safe_str(structured.get("Attribute", ""))
    diam = _safe_str(structured.get("Diameter", ""))
    year = _safe_str(structured.get("Year", ""))
    tsbc = _safe_str(structured.get("Technical Safety BC", ""))
    desc = _safe_str(structured.get("Description", ""))
    if not desc.strip():
        desc = _compute_description(ag, ubc, _safe_str(structured.get("Application", "")))
    approved_flag = '1' if _safe_str(structured.get("Approved", "")) == "True" else '0'

    return {
        "QR Code": _safe_str(qr),
        "Building": _safe_str(building),
        "Manufacturer": manuf,
        "Model": model,
        "Serial": serial,
        "UBC Tag": ubc,
        "Asset Group": ag,
        "Attribute": attr,
        "Description": desc,
        "Diameter": diam,
        "Year": year,
        "Technical Safety BC": tsbc,
        "Approved": approved_flag,
    }


def upsert_sdi_dataset(doc_id: str, structured: dict):
    if not _connectable():
        return
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=?', ("sdi_dataset",))
            if not cur.fetchone():
                print('⚠️ Table "sdi_dataset" not found; skipping sync.')
                return

            cols = _table_columns(conn, "sdi_dataset")
            payload = _build_sdi_row(doc_id, structured)
            if not payload:
                print(f"⚠️ Could not build sdi_dataset row for {doc_id}")
                return

            if "QR Code" not in cols:
                print('⚠️ "sdi_dataset" lacks "QR Code" column; cannot upsert key. Skipping.')
                return

            filtered = {k: v for k, v in payload.items() if k in cols}
            set_cols = [c for c in filtered.keys() if c != "QR Code"]
            if not set_cols:
                # If only the key is present, we may still need to INSERT it
                pass

            # Use composite key for upsert if "Building" column exists
            key_cols = ["QR Code"]
            if "Building" in cols:
                key_cols.append("Building")

            where_clause = " AND ".join(f'"{k}"=?' for k in key_cols)
            update_sql = f'UPDATE "sdi_dataset" SET {", ".join([f""""{c}"=? """ for c in set_cols])} WHERE {where_clause}'
            update_vals = [filtered[c] for c in set_cols] + [filtered[k] for k in key_cols]
            
            # Only run update if there are columns to set
            if set_cols:
                cur.execute(update_sql, update_vals)

            if cur.rowcount == 0:
                insert_cols = list(filtered.keys())
                placeholders = ",".join(["?"] * len(insert_cols))
                insert_sql = f'INSERT INTO "sdi_dataset" ({",".join(f"""\"{c}\"""" for c in insert_cols)}) VALUES ({placeholders})'
                cur.execute(insert_sql, [filtered[c] for c in insert_cols])

            conn.commit()
    except Exception as e:
        print(f"⚠️ Failed to upsert sdi_dataset for {doc_id}: {e}")


# ===== Load JSONs for dashboard =====
def load_json_items():
    items = []
    for filename in os.listdir(JSON_DIR):
        if not filename.endswith(".json") or filename.endswith("_raw_ocr.json"):
            continue

        if not _is_bf_filename(filename):
            continue
        
        m = JSON_NAME_RE.match(filename)
        if not m:
            continue

        qr, asset_type_mid, building = m.groups()
        doc_id = filename[:-5]

        try:
            with open(os.path.join(JSON_DIR, filename), 'r', encoding='utf-8') as f:
                raw = json.load(f)

            data = raw.get("structured_data") or {}
            if not isinstance(data, dict):
                print(f"⚠️ Skipped {filename}: 'structured_data' is not a dict")
                continue

            # Ensure keys
            data.setdefault("Manufacturer", "")
            data.setdefault("Model", "")
            data.setdefault("Serial Number", "")
            data.setdefault("UBC Tag", "")
            data.setdefault("Asset Group", "")
            data.setdefault("Attribute", "BackflowDevice")
            data.setdefault("Application", "")
            data.setdefault("Flagged", "false")
            data.setdefault("Approved", "")
            data.setdefault("Diameter", "")
            data.setdefault("Year", "")
            data.setdefault("Technical Safety BC", "")

            # Derived Description
            if not (data.get("Description") or "").strip():
                data["Description"] = _compute_description(
                    data.get("Asset Group"),
                    data.get("UBC Tag"),
                    data.get("Application", "")
                )

            # Missed photo rule: need >= 2 of 3
            present = 0
            exists_map = {}
            for tag in SEQ_CHECK:
                exists_map[tag] = bool(find_image(qr, building, "BF", tag))
                if exists_map[tag]:
                    present += 1
            missing_tags = [tag for tag in SEQ_CHECK if not exists_map[tag]]
            friendly_map = {'-0': 'Asset Plate/Label', '-1': 'Asset Plate/Label (Optional)', '-2': 'Main Asset'}
            missing_friendly = ", ".join(friendly_map.get(tag, tag) for tag in missing_tags)

            missed_photo = present < 2
            photos_summary = f"{present}/3"

            items.append({
                "doc_id": doc_id,
                "qr_code": qr,
                "building": building,
                "asset_type": "BF",
                "Flagged": data.get("Flagged", "false"),
                "Approved": data.get("Approved", ""),
                "Modified": raw.get("modified", False),
                "Missed Photo": "YES" if missed_photo else "NO",
                "Missing List": missing_friendly,
                "Photos Summary": photos_summary,
                **data
            })
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
    return items


@app.route("/")
def index():
    flagged_filter = request.args.get("flagged")
    modified_filter = request.args.get("modified")
    missed_filter = request.args.get("missed")

    all_data = load_json_items()

    count_flagged = sum(1 for item in all_data if item.get("Flagged") == "true")
    count_modified = sum(1 for item in all_data if item.get("Modified"))
    count_missed = sum(1 for item in all_data if item.get("Missed Photo") == "YES")

    data = all_data
    if flagged_filter == "true" and modified_filter == "true":
        data = [item for item in data if item.get("Flagged") == "true" and item.get("Modified")]
    elif flagged_filter == "true":
        data = [item for item in data if item.get("Flagged") == "true"]
    elif modified_filter == "true":
        data = [item for item in data if item.get("Modified")]

    if missed_filter == "true":
        data = [item for item in data if item.get("Missed Photo") == "YES"]

    return render_template(
        "dashboard.html",
        data=data,
        warn_missing=True,
        flagged_filter=flagged_filter,
        modified_filter=modified_filter,
        missed_filter=missed_filter,
        count_flagged=count_flagged,
        count_modified=count_modified,
        count_missed=count_missed
    )


@app.route("/review/<doc_id>")
def review(doc_id):
    json_path = os.path.join(JSON_DIR, f"{doc_id}.json")
    if not os.path.exists(json_path):
        return "Not found", 404

    m = JSON_NAME_RE.match(f"{doc_id}.json")
    if not m:
        return "Bad ID", 400

    qr, asset_type_mid, building = m.groups()
    if asset_type_mid.upper() != "BF":
        return "Not BF asset", 404

    with open(json_path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)

    data = loaded.get("structured_data", {}) or {}
    data.setdefault("Asset Group", "")
    data.setdefault("Attribute", "BackflowDevice")
    data.setdefault("Application", "")
    data.setdefault("UBC Tag", "")
    data.setdefault("Approved", "")
    data.setdefault("Diameter", "")
    data.setdefault("Year", "")
    data.setdefault("Technical Safety BC", "")

    if not (data.get("Description") or "").strip():
        data["Description"] = _compute_description(
            data.get("Asset Group"),
            data.get("UBC Tag"),
            data.get("Application", "")
        )

    images = {}
    for tag in SEQ_SHOW:
        filename = find_image(qr, building, "BF", tag)
        images[tag] = {"exists": bool(filename), "url": url_for('serve_image', filename=filename) if filename else None}

    asset_group_options = get_asset_group_options()
    attribute_options   = get_attribute_options()
    application_options = get_application_options()

    return render_template(
        "review.html",
        doc_id=doc_id,
        qr_code=qr,
        building=building,
        asset_type="BF",
        data=data,
        images=images,
        asset_group_options=asset_group_options,
        attribute_options=attribute_options,
        application_options=application_options
    )


@app.route("/review/<doc_id>", methods=["POST"])
def save_review(doc_id):
    json_path = os.path.join(JSON_DIR, f"{doc_id}.json")
    if not os.path.exists(json_path):
        return "Not found", 404

    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    structured = json_data.get("structured_data", {})
    if not isinstance(structured, dict):
        structured = {}
        json_data["structured_data"] = structured

    # Ensure keys exist
    structured.setdefault("Asset Group", "")
    structured.setdefault("Attribute", "BackflowDevice")
    structured.setdefault("Application", "")
    structured.setdefault("UBC Tag", "")
    structured.setdefault("Approved", "")
    structured.setdefault("Flagged", "false")
    structured.setdefault("Diameter", "")
    structured.setdefault("Year", "")
    structured.setdefault("Technical Safety BC", "")
    structured.setdefault("Description", "")

    # Update Flagged
    new_flagged = "true" if request.form.get("Flagged") == "on" else "false"
    if structured.get("Flagged", "false") != new_flagged:
        json_data["modified"] = True
    structured["Flagged"] = new_flagged

    # Update fields from form (respect user Description)
    for field in list(structured.keys()):
        if field in ("Flagged", "Approved"):
            continue
        form_value = request.form.get(field, "")
        if structured.get(field, "") != form_value:
            json_data["modified"] = True
        structured[field] = form_value

    # Capture any brand-new fields (skip navigation/query helper fields)
    for field, form_value in request.form.items():
        if field in ("Flagged", "dashboard_query", "_nav", "action"):
            continue
        if field not in structured:
            structured[field] = form_value
            json_data["modified"] = True

    # Compute default Description if blank
    desc_from_form = (request.form.get("Description", "") or "").strip()
    if not desc_from_form:
        auto_desc = _compute_description(
            structured.get("Asset Group"),
            structured.get("UBC Tag"),
            structured.get("Application", "")
        )
        if (structured.get("Description") or "").strip() != auto_desc:
            structured["Description"] = auto_desc
            json_data["modified"] = True

    # Persist JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    # Sync to sdi_dataset
    try:
        upsert_sdi_dataset(doc_id, structured)
    except Exception as e:
        print(f"⚠️ sdi_dataset sync failed on save for {doc_id}: {e}")

    # Next/Prev navigation within BF-only list
    all_files = sorted(
        f for f in os.listdir(JSON_DIR)
        if f.endswith(".json")
        and not f.endswith("_raw_ocr.json")
        and JSON_NAME_RE.match(f)
        and JSON_NAME_RE.match(f).groups()[1].upper() == "BF"
    )
    current_name = f"{doc_id}.json"
    try:
        current_index = all_files.index(current_name)
    except ValueError:
        dash_q = request.form.get("dashboard_query", "")
        if dash_q.startswith("?"):
            return redirect(url_for("index") + dash_q)
        return redirect(url_for("index"))

    # -------- Navigation decision (reads _nav from submit button) --------
    nav = request.form.get("_nav") or request.form.get("action") or "save"
    if nav == "save_next" and current_index + 1 < len(all_files):
        next_doc = all_files[current_index + 1][:-5]
        return redirect(url_for("review", doc_id=next_doc))
    elif nav == "save_prev" and current_index > 0:
        prev_doc = all_files[current_index - 1][:-5]
        return redirect(url_for("review", doc_id=prev_doc))

    dash_q = request.form.get("dashboard_query", "")
    if (dash_q or "").startswith("?"):
        return redirect(url_for("index") + dash_q)
    return redirect(url_for("index"))


@app.route("/toggle_approved/<doc_id>", methods=["POST"])
def toggle_approved(doc_id):
    """
    Toggle Approved between '' (False) and 'True' (True),
    persist to file, upsert QR_codes, and sync sdi_dataset.
    """
    json_path = os.path.join(JSON_DIR, f"{doc_id}.json")
    if not os.path.exists(json_path):
        return jsonify({"success": False, "error": "Not found"}), 404

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        structured = json_data.get("structured_data", {})
        if not isinstance(structured, dict):
            structured = {}
            json_data["structured_data"] = structured

        current = structured.get("Approved", "")
        structured["Approved"] = "True" if current == "" else ""
        json_data["structured_data"] = structured

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        parsed = parse_doc_id(doc_id)
        if parsed:
            qr, asset_type_mid, _building = parsed
            if asset_type_mid.upper() == "BF":
                upsert_approved_in_db(qr, structured["Approved"] == "True")

        try:
            upsert_sdi_dataset(doc_id, structured)
        except Exception as e:
            print(f"⚠️ sdi_dataset sync failed on toggle for {doc_id}: {e}")

        return jsonify({"success": True, "new_value": structured["Approved"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _quote(name: str) -> str:
    """Helper for safe SQL table/column name quoting."""
    return f'"{name}"'.replace('""', '"')


@app.route("/check_sdi/<qr_code>")
def check_sdi(qr_code):
    """
    Checks if a QR code exists in the sdi_print_out table to prevent
    un-approving an asset that has already been exported to Planon.
    """
    if not _connectable():
        return jsonify({"error": "Database not accessible"}), 500

    sdi_print_out_table = "sdi_print_out"
    qr_col = "QR Code"

    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            query = f"SELECT 1 FROM {_quote(sdi_print_out_table)} WHERE {_quote(qr_col)} = ? LIMIT 1"
            cur.execute(query, (qr_code,))
            result = cur.fetchone()
            return jsonify({"exists": result is not None})
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return jsonify({"exists": False})
        
        print(f"!! DB Operational Error in /check_sdi (BF): {e}")
        return jsonify({"error": f"Database query failed: {e}"}), 500
    except Exception as e:
        print(f"!! UNEXPECTED ERROR in /check_sdi (BF): {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMG_DIR, filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003, debug=True)
