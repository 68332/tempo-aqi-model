import csv
import gzip
import io
import os
import requests
import xml.etree.ElementTree as ET
import datetime as dt

BASE = "https://openaq-data-archive.s3.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# -------- S3 列表工具 --------
def list_s3(prefix, delimiter=None):
    """回傳 (keys, prefixes)
    - keys: 這層底下的物件完整 key 清單
    - prefixes: 子資料夾（以 delimiter 分隔）的前綴清單
    """
    params = {"list-type": "2", "prefix": prefix}
    if delimiter:
        params["delimiter"] = delimiter
    r = requests.get(BASE, params=params, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    keys = [c.find("s3:Key", NS).text for c in root.findall("s3:Contents", NS)]
    return keys

# -------- header --------
HEADER = ["date", "pm25", "pm10", "o3", "no2", "so2", "co"]

def ensure_out_csv(path: str):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(HEADER)

def load_out_as_map(path: str):
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            date_key = row["date"]
            data[date_key] = {
                "pm25": row.get("pm25") or "",
                "pm10": row.get("pm10") or "",
                "o3":   row.get("o3") or "",
                "no2":  row.get("no2") or "",
                "so2":  row.get("so2") or "",
                "co":   row.get("co") or "",
            }
    return data

def write_out_from_map(path: str, data_map: dict):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for date_key in sorted(data_map.keys()):
            row = data_map[date_key]
            w.writerow([
                date_key,
                row.get("pm25", ""),
                row.get("pm10", ""),
                row.get("o3", ""),
                row.get("no2", ""),
                row.get("so2", ""),
                row.get("co", ""),
            ])

def date_to_hour(date_str: str) -> str:
    s = (date_str or "").strip()
    if not s:
        return ""
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:
            d = dt.datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return s[:13] + ":00:00"

    # 把分鐘、秒、微秒歸零
    d = d.replace(minute=0, second=0, microsecond=0)

    # 如果原字串有時區，就保留 tzinfo
    if d.tzinfo is not None:
        return d.isoformat()
    else:
        return d.strftime("%Y-%m-%dT%H:00:00")

def update_map(out_map: dict, day_key: str, pollutant: str, value: str):
    if not day_key:
        return
    if day_key not in out_map:
        out_map[day_key] = {"pm25": "", "pm10": "", "o3": "", "no2": "", "so2": "", "co": ""}
    if pollutant in out_map[day_key]:
        # If there's already a value, we keep the first one we encountered
        out_map[day_key][pollutant] = value 

# -------- 單檔下載並合併進 out.csv 寬表 --------
def process_file_to_wide_table(key: str, out_csv: str, out_map: dict):
    url = f"{BASE}/{key}"
    print(f"[READ ] {url}")

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()

        if key.endswith(".gz"):
            reader = io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw), encoding="utf-8", newline="")
        else:
            reader = io.TextIOWrapper(r.raw, encoding="utf-8", newline="")

        csv_reader = csv.reader(reader)
        header = next(csv_reader, None)
        if header is None:
            print("[WARN ] empty file, skip")
            return

        # 優先用欄名找 index；若沒表頭或找不到，就 fallback 到你原本的索引
        def find_idx(names, fallback):
            if header:
                for n in names:
                    if n in header:
                        return header.index(n)
            return fallback

        idx_parameter = find_idx(["parameter", "pollutant"], 6)    # 第7欄
        idx_value     = find_idx(["value", "measurement"], 8)      # 第9欄
        idx_datetime  = find_idx(["date.local", "date.utc", "datetime", "timestamp"], 3)  # 第4欄

        for cols in csv_reader:
            if not cols:
                continue
            try:
                pollutant = cols[idx_parameter].strip().lower()
                if pollutant not in {"pm25", "pm10", "o3", "no2", "so2", "co"}:
                    continue

                value_str = cols[idx_value].strip()
                # （可選）嘗試轉 float 以過濾奇怪字串
                try:
                    v = float(value_str)
                    value_str = f"{v:.10g}"
                except Exception:
                    # 無法轉數字就原樣放入
                    pass

                day_key = date_to_hour(cols[idx_datetime])
                print("day_key:", day_key)
                update_map(out_map, day_key, pollutant, value_str)

            except IndexError:
                # 欄位不齊，跳過
                continue

    print(f"[OK   ] merged {key} -> {out_csv}")

# -------- 下載整個 location 的資料並寫入 out.csv（寬表） --------
def fetch_location(locationid, year_from=2020, year_to=2022, outdir="data"):
    outdir = f"data_{locationid}_{year_from}-{year_to}"
    out_csv = os.path.join(outdir, f"locationid{locationid}_{year_from}-{year_to}.csv")

    ensure_out_csv(out_csv)
    out_map = load_out_as_map(out_csv)

    for year in range(year_from, year_to + 1):
        for m in range(6, 13):
            mm = f"{m:02d}"
            month_prefix = f"records/csv.gz/locationid={locationid}/year={year}/month={mm}/"

            # 第一次：列出該月底下的「天」prefix
            day_prefixes = list_s3(month_prefix, delimiter="/")
            if not day_prefixes:
                print(f"[info ] no data for {locationid} {year}-{mm}")
                continue

            for day_prefix in day_prefixes:
                print(f"[info ] processing day prefix: {day_prefix}")
                process_file_to_wide_table(day_prefix, out_csv, out_map)
                write_out_from_map(out_csv, out_map)

    # end of all, write out the final map
    write_out_from_map(out_csv, out_map)
    print(f"[DONE ] wrote wide table -> {out_csv}")

if __name__ == "__main__":
    fetch_location(locationid=221, year_from=2025, year_to=2025)
