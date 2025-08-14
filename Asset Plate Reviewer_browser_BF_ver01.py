import os
import json
import re
import sqlite3
from functools import lru_cache
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify

app = Flask(
    __name__,
    template_folder=r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\Git_control\Asset_plate_review_BF\review_asset_templates",
    static_folder=r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\Git_control\Asset_plate_review_BF\review_asset_templates\static"
)

# --- Paths ---
JSON_DIR = r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\API\Output_jason_api"
IMG_DIR  = r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\Capture_photos_upload"

# --- SQLite DB (for dropdown options) ---
DB_PATH = r"S:\MaintOpsPlan\AssetMgt\Asset Management Process\Database\8. New Assets\QR_code_project\asset_capture_app\data\QR_codes.db"

# Tables/columns (only used where relevant)
ASSET_GROUP_TABLE = "Asset_Group"
ASSET_GROUP_COL   = "name"
ATTRIBUTE_TABLE   = "Attribute"       # not used directly since we lock Attribute to BackflowDevice

# --- Application table autodetection ---
APPLICATION_TABLE_CANDIDATES = [
    "bf_applicaton_type",   # as originally provided
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


def find_image(qr: str, building: str, asset_type_mid: str, seq_tag: str):
    """
    Find image by pattern: '<QR> <Building> <TYPE> - <seq>.<ext>'.
    TYPE is 'BF' per current scope/spec.
    """
    seq = seq_tag.replace('-', '').strip()
    base = f"{qr} {building} {asset_type_mid} - {seq}"
    for ext in VALID_IMAGE_EXTS:
        candidate = os.path.join(IMG_DIR, base + ext)
        if os.path.exists(candidate):
            return os.path.basename(candidate)
    return None


@lru_cache(maxsize=1)
def _connectable():
    """Check DB path once (cached)."""
    return os.path.exists(DB_PATH)


def _fetch_column_values(table: str, col: str):
    """Return sorted unique non-empty strings for dropdowns."""
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
    """Restrict Asset Group dropdown to the two Backflow groups."""
    all_opts = _fetch_column_values(ASSET_GROUP_TABLE, ASSET_GROUP_COL)
    allowed = {"Primary Backflow Devices", "Secondary Backflow Devices"}
    return [opt for opt in all_opts if opt in allowed]


def get_attribute_options():
    """Restrict Attribute dropdown to BackflowDevice only."""
    return ["BackflowDevice"]


# --- Robust Application loader helpers ---
def _table_exists(conn, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    return cur.fetchone() is not None


def _text_columns(conn, table_name: str):
    """
    Return a list of column names whose declared type looks text-like.
    Falls back to *all* columns if types aren't declared.
    """
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
    """
    Load Application dropdown values:
      - Try multiple table names.
      - Introspect columns and use the first text-like column with data.
    """
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
    """
    Build Description as:
      BFP-<Application + space><First word of Asset Group + space>BFP

    Example:
      Application='DCW', Asset Group='Secondary Backflow Devices'
      -> 'BFP-DCW Secondary BFP'

    Note: ubc_tag kept in signature for compatibility; not used in this format.
    """
    prefix = "BFP-"
    suffix = "BFP"

    app_part = (application or "").strip()
    if app_part:
        app_part += " "

    first_word_group = ""
    if asset_group:
        first_word_group = asset_group.strip().split()[0] + " "

    return f"{prefix}{app_part}{first_word_group}{suffix}".strip()


def load_json_items():
    items = []
    for filename in os.listdir(JSON_DIR):
        if not filename.endswith(".json") or filename.endswith("_raw_ocr.json"):
            continue

        m = JSON_NAME_RE.match(filename)
        if not m:
            continue

        qr, asset_type_mid, building = m.groups()

        # ✅ Only BF assets
        if asset_type_mid.upper() != "BF":
            continue

        doc_id = filename[:-5]  # strip ".json"

        try:
            with open(os.path.join(JSON_DIR, filename), 'r', encoding='utf-8') as f:
                raw = json.load(f)

            data = raw.get("structured_data") or {}
            if not isinstance(data, dict):
                print(f"⚠️ Skipped {filename}: 'structured_data' is not a dict")
                continue

            # Ensure keys exist to keep them visible/editable
            data.setdefault("Manufacturer", "")
            data.setdefault("Model", "")
            data.setdefault("Serial Number", "")
            data.setdefault("UBC Tag", "")
            data.setdefault("Asset Group", "")
            data.setdefault("Attribute", "BackflowDevice")   # default value
            data.setdefault("Application", "")               # new field
            data.setdefault("Flagged", "false")
            data.setdefault("Approved", "")                  # default blank (False)
            data.setdefault("Diameter", "")                  # new field

            # Derived: Description from Application + Asset Group
            # Only compute if missing/blank to preserve manual overrides
            if not (data.get("Description") or "").strip():
                data["Description"] = _compute_description(
                    data.get("Asset Group"),
                    data.get("UBC Tag"),
                    data.get("Application", "")
                )

            # ✅ Missed photo rule: require at least 2 of 3 present
            present = 0
            exists_map = {}
            for tag in SEQ_CHECK:
                exists_map[tag] = bool(find_image(qr, building, "BF", tag))
                if exists_map[tag]:
                    present += 1

            missing_tags = [tag for tag in SEQ_CHECK if not exists_map[tag]]
            friendly_map = {
                '-0': 'Asset Plate/Label',
                '-1': 'Asset Plate/Label (Optional)',
                '-2': 'Main Asset'
            }
            missing_friendly = ", ".join(friendly_map.get(tag, tag) for tag in missing_tags)

            missed_photo = present < 2  # only flag if < 2 present
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

    all_data = load_json_items()  # BF-only

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
    data.setdefault("Attribute", "BackflowDevice")  # default value for Attribute
    data.setdefault("Application", "")              # ensure presence for edit
    data.setdefault("UBC Tag", "")
    data.setdefault("Approved", "")
    data.setdefault("Diameter", "")

    # Derived Description (compute only if blank to preserve manual values)
    if not (data.get("Description") or "").strip():
        data["Description"] = _compute_description(
            data.get("Asset Group"),
            data.get("UBC Tag"),
            data.get("Application", "")
        )

    # Build image map (no -3)
    images = {}
    for tag in SEQ_SHOW:
        filename = find_image(qr, building, "BF", tag)
        if filename:
            images[tag] = {"exists": True, "url": url_for('serve_image', filename=filename)}
        else:
            images[tag] = {"exists": False, "url": None}

    # Dropdown options from DB
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
    structured.setdefault("Description", "")  # ensure key exists

    # Update Flagged
    new_flagged = "true" if request.form.get("Flagged") == "on" else "false"
    if structured.get("Flagged", "false") != new_flagged:
        json_data["modified"] = True
    structured["Flagged"] = new_flagged

    # Update fields from form (now we DO allow Description to be user-edited)
    for field in list(structured.keys()):
        if field in ("Flagged", "Approved"):
            continue
        form_value = request.form.get(field, "")
        if structured.get(field, "") != form_value:
            json_data["modified"] = True
        structured[field] = form_value

    # Capture any brand-new fields (future-proof)
    for field, form_value in request.form.items():
        if field in ("Flagged", "action", "dashboard_query"):
            continue
        if field not in structured:
            structured[field] = form_value
            json_data["modified"] = True

    # If Description came in blank, compute default; otherwise respect user text
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

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    # Next/Prev navigation stays within review context (BF-only order)
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

    action = request.form.get("action")
    if action == "save_next" and current_index + 1 < len(all_files):
        next_doc = all_files[current_index + 1][:-5]
        return redirect(url_for("review", doc_id=next_doc))
    elif action == "save_prev" and current_index > 0:
        prev_doc = all_files[current_index - 1][:-5]
        return redirect(url_for("review", doc_id=prev_doc))

    dash_q = request.form.get("dashboard_query", "")
    if (dash_q or "").startswith("?"):
        return redirect(url_for("index") + dash_q)

    return redirect(url_for("index"))


@app.route("/toggle_approved/<doc_id>", methods=["POST"])
def toggle_approved(doc_id):
    """Toggle Approved between '' (False) and 'True' (True) and persist to file."""
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

        return jsonify({"success": True, "new_value": structured["Approved"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMG_DIR, filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
