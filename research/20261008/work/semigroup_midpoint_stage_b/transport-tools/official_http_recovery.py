"""Use the public BSCC web app's documented-in-source HTTP calls.

Credentials and session cookies remain in this process; output contains neither.
Run as a JSON-lines worker so a successful session can serve multiple reads.
"""
from pathlib import Path
import hashlib
import http.cookiejar
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://file.blsc.cn/api"
CLOUD = "https://cloud.blsc.cn"
USER = "https://user.blsc.cn/user/api"
jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/140 Safari/537.36",
    "Origin": CLOUD,
    "Referer": CLOUD + "/",
    "Accept": "application/json, text/javascript, */*",
}
emit_lock = threading.Lock()
event_log = None


def emit(value):
    line = json.dumps(value, ensure_ascii=False)
    with emit_lock:
        print(line, flush=True)
        if event_log is not None:
            with event_log.open("a") as handle:
                handle.write(line + "\n")


def request(url, data=None, extra_headers=None):
    req_headers = dict(headers)
    if extra_headers:
        req_headers.update(extra_headers)
    if data is not None:
        data = urllib.parse.urlencode(data).encode()
        req_headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, headers=req_headers)
    try:
        return opener.open(req, timeout=35)
    except urllib.error.HTTPError as exc:
        return exc


def inspect_json(response):
    raw = response.read(2 * 1024 * 1024)
    try:
        parsed = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        parsed = None
    return parsed, {
        "http_status": response.status,
        "response_bytes": len(raw),
        "content_type": response.headers.get("Content-Type"),
        "json_type": type(parsed).__name__,
        "json_keys": list(parsed) if isinstance(parsed, dict) else None,
        "service_code": parsed.get("code") if isinstance(parsed, dict) else None,
    }


def login():
    credential_path = Path("/Users/Apple_501/.codex/skills/beijing-supercompute/references/login-credentials.md")
    raw = credential_path.read_text()
    account = re.search(r"^account=(.+)$", raw, re.M).group(1).strip()
    password = re.search(r"^password=(.+)$", raw, re.M).group(1).strip()
    # The official login component calls spark-md5.hash on the entered password.
    form = dict(token_type="COOKIE", cookie_type="WAL", password=hashlib.md5(password.encode()).hexdigest(),
                third_party="SELF", email=account, phone="")
    response = request(USER + "/login?" + str(int(time.time() * 1000)), form)
    parsed, info = inspect_json(response)
    info["action"] = "login"
    info["cookie_count"] = len(jar)
    emit(info)
    if response.status != 200:
        return False
    query = request(USER + "/query?" + urllib.parse.urlencode(dict(token_type="COOKIE", winfo="true")))
    parsed, info = inspect_json(query)
    info["action"] = "verify_login"
    info["cookie_count"] = len(jar)
    identity_matches = isinstance(parsed, dict) and parsed.get("email") == account and bool(parsed.get("userId"))
    info["identity_matches"] = identity_matches
    emit(info)
    return query.status == 200 and bool(jar) and identity_matches


def run_command(command):
    action = command["action"]
    if action == "quit":
        return False
    if action == "batch_download":
        jobs = command["downloads"]
        if any(job.get("action") != "download" for job in jobs):
            raise ValueError("Batch accepts only file downloads")
        with ThreadPoolExecutor(max_workers=min(int(command.get("workers", 4)), 6)) as executor:
            futures = {executor.submit(run_command, job): job for job in jobs}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    emit({"status": "COMMAND_ERROR", "file": Path(futures[future]["output"]).name,
                          "error_type": type(exc).__name__})
        emit({"action": action, "status": "FINISHED", "requested_files": len(jobs)})
        return True
    if action in {"list", "json", "cloud_json"}:
        endpoint = command.get("endpoint", "file/manager/ls")
        base = command.get("base", CLOUD + "/api" if action == "cloud_json" else BASE)
        params = command.get("params", {"path": command.get("path", "")})
        url = base + "/" + endpoint + "?" + urllib.parse.urlencode(params)
        parsed, info = inspect_json(request(url))
        info["action"] = action
        # Only emit explicit file metadata, never a raw authentication response.
        if action == "list":
            rows = parsed if isinstance(parsed, list) else []
            if isinstance(parsed, dict):
                info["list_fields"] = {k: len(v) for k, v in parsed.items() if isinstance(v, list)}
                for key in ("data", "files", "list", "items", "result"):
                    if isinstance(parsed.get(key), list):
                        rows = parsed[key]
                        break
            info["items"] = [{k: v for k, v in row.items() if k in {"name", "fileName", "size", "path", "type", "isDir"}} for row in rows if isinstance(row, dict)]
            if rows:
                info["first_item_keys"] = list(rows[0]) if isinstance(rows[0], dict) else []
        if "save_public_json" in command and info["http_status"] == 200:
            # Caller may save only file/connection metadata, never user/session responses.
            if endpoint.startswith(("user", "login", "query", "auth")):
                raise ValueError("Authentication response storage is forbidden")
            target = Path(command["save_public_json"])
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x") as handle:
                json.dump(parsed, handle, ensure_ascii=False, indent=2)
            info["saved"] = str(target)
        emit(info)
    elif action == "download":
        target = Path(command["output"])
        partial = target.with_suffix(target.suffix + ".partial")
        expected_sha = command.get("sha256")
        expected_size = int(command["bytes"])
        if target.exists():
            digest = hashlib.sha256()
            with target.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            matched = target.stat().st_size == expected_size and (expected_sha is None or actual == expected_sha)
            status = ("EXISTING_VERIFIED" if expected_sha else "EXISTING_SIZE_MATCH_REQUIRES_PARENT_SHA") if matched else "EXISTING_MISMATCH"
            emit({"action": action, "status": status, "file": target.name, "bytes": target.stat().st_size, "sha256": actual})
            return True
        if partial.exists() and not command.get("resume", False):
            emit({"action": action, "status": "PARTIAL_EXISTS", "file": partial.name})
            return True
        base = command.get("base", BASE)
        endpoint = command.get("endpoint", "file/stream/download")
        url = base + "/" + endpoint + "?" + urllib.parse.urlencode({"path": command["path"]})
        target.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        total = 0
        for attempt in range(1, 9):
            digest = hashlib.sha256()
            if partial.exists():
                with partial.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                total = partial.stat().st_size
            if total > expected_size:
                raise ValueError("Partial exceeds recorded archive size")
            if total == expected_size:
                break
            extra = {"Accept": "application/octet-stream"}
            if total:
                extra["Range"] = "bytes=%s-%s" % (total, expected_size - 1)
            response = None
            try:
                response = request(url, extra_headers=extra)
                if response.status not in (200, 206):
                    _, info = inspect_json(response)
                    info.update(action=action, status="HTTP_ERROR", file=target.name, attempt=attempt)
                    emit(info)
                    if info.get("service_code") == 20002:
                        # The API rejects an invalid byte-range form; this is
                        # a request error, so repeating it cannot help.
                        return True
                    if response.status in (401, 403) or response.status not in (408, 429, 500, 502, 503, 504):
                        return True
                    time.sleep(min(attempt * 2, 10))
                    continue
                content_range = response.headers.get("Content-Range")
                if total:
                    expected_range = "bytes %s-%s/%s" % (total, expected_size - 1, expected_size)
                    # This API returns HTTP 200 even for exact byte ranges.
                    # Require the exact Content-Range before appending anything.
                    if content_range != expected_range:
                        raise ValueError("Server did not acknowledge the exact resume offset")
                    emit({"action": action, "status": "RESUME_RANGE_ACKNOWLEDGED", "file": target.name,
                          "offset": total, "prefix_sha256": digest.hexdigest(), "content_range": content_range,
                          "host": urllib.parse.urlparse(base).hostname})
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != expected_size - total:
                    raise ValueError("Response length differs from recorded remaining size")
                last_emit = time.monotonic()
                with partial.open("ab" if partial.exists() else "xb") as handle:
                    try:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            if total + len(chunk) > expected_size:
                                raise ValueError("Response exceeded recorded archive size")
                            handle.write(chunk)
                            digest.update(chunk)
                            total += len(chunk)
                            if time.monotonic() - last_emit > 15:
                                emit({"action": action, "status": "PROGRESS", "file": target.name,
                                      "bytes": total, "expected_bytes": expected_size})
                                last_emit = time.monotonic()
                    finally:
                        handle.flush()
                        os.fsync(handle.fileno())
                if total == expected_size:
                    break
                emit({"action": action, "status": "EARLY_EOF_RETAINED", "file": target.name,
                      "bytes": total, "attempt": attempt})
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                emit({"action": action, "status": "STREAM_INTERRUPTED_RETAINED", "file": target.name,
                      "bytes": total, "attempt": attempt, "error_type": type(exc).__name__})
                time.sleep(min(attempt * 2, 10))
            finally:
                if response is not None:
                    response.close()
        actual_sha = digest.hexdigest()
        matched = total == expected_size and (expected_sha is None or actual_sha == expected_sha)
        if matched:
            partial.rename(target)
        status = (("VERIFIED" if expected_sha else "SIZE_MATCH_REQUIRES_PARENT_SHA") if matched
                  else "INCOMPLETE_RETAINED" if total < expected_size else "MISMATCH")
        emit({"action": action, "status": status, "file": target.name, "bytes": total,
              "sha256": actual_sha, "elapsed_seconds": round(time.monotonic() - started, 3)})
    else:
        raise ValueError("Unknown action")
    return True


if __name__ == "__main__":
    try:
        if len(sys.argv) == 2:
            event_log = Path(sys.argv[1])
            event_log.parent.mkdir(parents=True, exist_ok=True)
            with event_log.open("x"):
                pass
        authenticated = login()
        if not authenticated:
            emit({"status": "LOGIN_NOT_VERIFIED"})
            sys.exit(2)
        emit({"status": "WORKER_READY"})
        for line in sys.stdin:
            try:
                if not run_command(json.loads(line)):
                    break
            except Exception as exc:
                emit({"status": "COMMAND_ERROR", "error_type": type(exc).__name__})
    except Exception as exc:
        emit({"status": "WORKER_ERROR", "error_type": type(exc).__name__})
        sys.exit(2)
