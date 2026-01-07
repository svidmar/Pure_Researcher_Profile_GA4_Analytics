#!/usr/bin/env python3
"""
A simple, interactive web application for fetching related persons from an organizational unit in Pure from the Pure API and analyzing it with Google Analytics data. Get organization totals or breakdown per person.
"""

import io
import os
import re
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

DEFAULT_PURE_BASE = "https://vbn.aau.dk"
DEFAULT_API_VERSION = "524"

FIELDS = [
    "uuid",
    "name.firstName",
    "name.lastName",
    "info.prettyURLIdentifiers",
]

PERSON_PROFILE_RE = re.compile(r"^/(da|en)/persons/[^/]+$")


def fetch_page(
    pure_base: str,
    api_version: str,
    unit_id: str,
    api_key: str,
    offset: int,
    size: int,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Fetch a single page of persons from Pure API"""
    url = f"{pure_base}/ws/api/{api_version}/organisational-units/{unit_id}/persons"
    headers = {"Accept": "application/json", "api-key": api_key}

    params = [("offset", str(offset)), ("size", str(size))]
    params += [("fields", f) for f in FIELDS]

    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, dict) or "items" not in data:
        raise ValueError(
            f"Unexpected response shape. Top-level keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )
    return data


def fetch_all_persons(
    pure_base: str,
    api_version: str,
    unit_id: str,
    api_key: str,
    page_size: int = 100,
    max_per_sec: float = 3.0,
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """Fetch all persons from Pure API with pagination"""
    first = fetch_page(
        pure_base=pure_base,
        api_version=api_version,
        unit_id=unit_id,
        api_key=api_key,
        offset=0,
        size=page_size,
    )

    count = int(first.get("count", 0))
    page_info = first.get("pageInformation") or {}
    size = int(page_info.get("size", page_size))
    offset = int(page_info.get("offset", 0))

    items = list(first.get("items") or [])

    if progress_callback:
        progress_callback(len(items), count)

    if count <= len(items):
        return items

    sleep_s = 1.0 / max_per_sec if max_per_sec and max_per_sec > 0 else 0.0
    next_offset = offset + size

    while next_offset < count:
        time.sleep(sleep_s)
        page = fetch_page(
            pure_base=pure_base,
            api_version=api_version,
            unit_id=unit_id,
            api_key=api_key,
            offset=next_offset,
            size=size,
        )
        page_items = page.get("items") or []
        if not page_items:
            break
        items.extend(page_items)

        if progress_callback:
            progress_callback(len(items), count)

        pi = page.get("pageInformation") or {}
        size = int(pi.get("size", size))
        next_offset = int(pi.get("offset", next_offset)) + size

    return items


def build_urls_with_types(
    uuid: str,
    pretty_ids: List[str],
    langs: List[str],
    template: str,
) -> List[Tuple[str, str, str, str]]:
    """
    Build URL variants for a person
    Returns list of tuples: (lang, identifier, identifier_type, url_path)
    """
    candidates: List[Tuple[str, str]] = [("uuid", uuid)]
    for pid in pretty_ids:
        candidates.append(("pretty", pid))

    seen = set()
    deduped: List[Tuple[str, str]] = []
    for id_type, ident in candidates:
        ident = str(ident).strip()
        if not ident:
            continue
        key = (id_type, ident)
        if key not in seen:
            seen.add(key)
            deduped.append((id_type, ident))

    rows: List[Tuple[str, str, str, str]] = []
    for lang in langs:
        for id_type, ident in deduped:
            url_path = template.format(lang=lang, identifier=ident)
            rows.append((lang, ident, id_type, url_path))
    return rows


def process_persons(persons: List[Dict[str, Any]], langs: List[str], template: str) -> pd.DataFrame:
    """Process persons data and create URL mapping DataFrame"""
    mapping_rows: List[Dict[str, Any]] = []

    for p in persons:
        uuid = str(p.get("uuid", "")).strip()
        first = ((p.get("name") or {}).get("firstName") or "").strip()
        last = ((p.get("name") or {}).get("lastName") or "").strip()
        name = (first + " " + last).strip()

        ids = (((p.get("info") or {}).get("prettyURLIdentifiers")) or [])
        if not isinstance(ids, list):
            ids = []

        pretty_ids: List[str] = []
        seen = set()
        for x in ids:
            s = str(x).strip()
            if s and s not in seen:
                pretty_ids.append(s)
                seen.add(s)

        urls = build_urls_with_types(
            uuid=uuid,
            pretty_ids=pretty_ids,
            langs=langs,
            template=template,
        )

        for lang, ident, id_type, url_path in urls:
            mapping_rows.append(
                {
                    "uuid": uuid,
                    "name": name,
                    "lang": lang,
                    "identifier": ident,
                    "identifier_type": id_type,
                    "url_path": url_path,
                }
            )

    return pd.DataFrame(mapping_rows)


def normalize_path(s: str) -> str:
    """Normalize URL path for matching"""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    if s.startswith("http://") or s.startswith("https://"):
        u = urlparse(s)
        s = u.path
    s = s.split("?", 1)[0]
    if len(s) > 1 and s.endswith("/"):
        s = s[:-1]
    return s


def read_ga4_csv(uploaded_file) -> pd.DataFrame:
    """
    Read GA4 CSV file with flexible separator detection
    """
    content = uploaded_file.getvalue().decode("utf-8-sig")

    for sep in (";", ","):
        try:
            df = pd.read_csv(io.StringIO(content), sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue

    raise ValueError("Could not parse GA4 CSV with ';' or ',' separators.")


def join_with_ga4(mapping_df: pd.DataFrame, ga4_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join Pure mapping with GA4 data and return results"""
    mapping_df["join_key"] = mapping_df["url_path"].apply(normalize_path)

    path_col = "Page path and screen class"
    views_col = "Views"
    users_col = "Active users"

    missing = [c for c in [path_col, views_col, users_col] if c not in ga4_df.columns]
    if missing:
        raise ValueError(f"GA4 CSV missing columns: {missing}. Found: {list(ga4_df.columns)}")

    ga4_df["join_key"] = ga4_df[path_col].apply(normalize_path)

    # Keep only actual person profile pages
    ga4_df = ga4_df[ga4_df["join_key"].apply(lambda x: bool(PERSON_PROFILE_RE.match(x)))].copy()

    ga4_agg = (
        ga4_df.groupby("join_key", as_index=False)
        .agg({views_col: "sum", users_col: "sum"})
        .rename(columns={views_col: "views", users_col: "active_users"})
    )

    merged = mapping_df.merge(ga4_agg, on="join_key", how="left")
    merged["views"] = merged["views"].fillna(0).astype(int)
    merged["active_users"] = merged["active_users"].fillna(0).astype(int)

    per_person = (
        merged.groupby(["uuid", "name"], as_index=False)
        .agg(
            views=("views", "sum"),
            active_users=("active_users", "sum"),
            url_variants=("join_key", "nunique"),
        )
        .sort_values(["views"], ascending=False)
    )

    section_total = pd.DataFrame([{
        "views_total": int(per_person["views"].sum()),
        "active_users_total": int(per_person["active_users"].sum()),
        "persons_count": int(per_person.shape[0]),
    }])

    mapped_keys = set(mapping_df["join_key"].dropna().unique())
    unmatched = ga4_agg[~ga4_agg["join_key"].isin(mapped_keys)].sort_values("views", ascending=False)

    return per_person, section_total, unmatched


def to_excel(per_person_df: pd.DataFrame, totals_df: pd.DataFrame) -> bytes:
    """Convert DataFrames to Excel file in memory"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        totals_df.to_excel(writer, sheet_name='Organization Total', index=False)
        per_person_df.to_excel(writer, sheet_name='Per Person', index=False)
    return output.getvalue()


def main():
    st.set_page_config(
        page_title="Pure + GA4 Analytics",
        layout="wide"
    )

    st.title("Pure Organization Analytics with GA4")
    st.markdown("A simple, interactive web application for fetching related persons from an organizational unit in Pure from the Pure API and analyzing it with Google Analytics data. Get organization totals or breakdown per person.")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        unit_id = st.text_input(
            "Organization Unit ID",
            placeholder="e.g., 12345678",
            help="The Pure organizational unit ID to fetch persons from"
        )

        api_key = st.text_input(
            "Pure API Key",
            type="password",
            value=os.getenv("PURE_API_KEY", ""),
            help="Your Pure API key (or set PURE_API_KEY environment variable)"
        )

        with st.expander("Advanced Settings"):
            pure_base = st.text_input("Pure Base URL", value=DEFAULT_PURE_BASE)
            api_version = st.text_input("API Version", value=DEFAULT_API_VERSION)
            langs = st.text_input("Languages (comma-separated)", value="da,en")
            profile_template = st.text_input("Profile URL Template", value="/{lang}/persons/{identifier}")

    # Main content
    if not unit_id:
        st.info("Please enter an Organization Unit ID in the sidebar to get started")
        return

    if not api_key:
        st.warning("Please provide a Pure API key in the sidebar")
        return

    # Step 1: Fetch persons from Pure
    st.header("Step 1: Fetch Persons from Pure")

    if "persons_data" not in st.session_state:
        st.session_state.persons_data = None
        st.session_state.mapping_df = None

    if st.button("Fetch Persons from Pure API", type="primary"):
        with st.spinner("Fetching persons from Pure API..."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                def progress_callback(current, total):
                    progress = current / total if total > 0 else 0
                    progress_bar.progress(progress)
                    status_text.text(f"Fetched {current} of {total} persons...")

                langs_list = [x.strip() for x in langs.split(",") if x.strip()] or ["da", "en"]

                persons = fetch_all_persons(
                    pure_base=pure_base,
                    api_version=api_version,
                    unit_id=unit_id,
                    api_key=api_key,
                    progress_callback=progress_callback
                )

                mapping_df = process_persons(persons, langs_list, profile_template)

                st.session_state.persons_data = persons
                st.session_state.mapping_df = mapping_df

                progress_bar.empty()
                status_text.empty()
                st.success(f"Successfully fetched {len(persons)} persons!")

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                return

    if st.session_state.persons_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Persons", len(st.session_state.persons_data))
        with col2:
            st.metric("URL Mappings", len(st.session_state.mapping_df))
        with col3:
            unique_names = st.session_state.mapping_df["name"].nunique()
            st.metric("Unique Names", unique_names)

        with st.expander("View Person Details"):
            st.dataframe(
                st.session_state.mapping_df[["name", "uuid", "identifier_type", "url_path"]].drop_duplicates(),
                use_container_width=True
            )

    # Step 2: Upload GA4 data and join
    st.header("Step 2: Upload and Join GA4 Data")

    if st.session_state.mapping_df is None:
        st.info("Please fetch persons data first before uploading GA4 data")
        return

    uploaded_file = st.file_uploader(
        "Upload GA4 Export CSV",
        type=["csv"],
        help="Export from Google Analytics with columns: 'Page path and screen class', 'Views', 'Active users'"
    )

    if uploaded_file is not None:
        try:
            ga4_df = read_ga4_csv(uploaded_file)
            st.success(f"GA4 file loaded: {len(ga4_df)} rows")

            with st.spinner("Joining data..."):
                per_person_df, totals_df, unmatched_df = join_with_ga4(
                    st.session_state.mapping_df.copy(),
                    ga4_df.copy()
                )

            # Display results
            st.header("Results")

            # Organization totals
            st.subheader("Organization Totals")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Views", f"{totals_df['views_total'].iloc[0]:,}")
            with col2:
                st.metric("Total Active Users", f"{totals_df['active_users_total'].iloc[0]:,}")
            with col3:
                st.metric("Persons with Data", totals_df['persons_count'].iloc[0])

            # Per person breakdown
            st.subheader("Per Person Breakdown")

            # Add filters
            col1, col2 = st.columns([3, 1])
            with col1:
                search_term = st.text_input("Search by name", "")
            with col2:
                min_views = st.number_input("Minimum views", min_value=0, value=0)

            filtered_df = per_person_df.copy()
            if search_term:
                filtered_df = filtered_df[filtered_df["name"].str.contains(search_term, case=False, na=False)]
            if min_views > 0:
                filtered_df = filtered_df[filtered_df["views"] >= min_views]

            st.dataframe(
                filtered_df.style.format({
                    "views": "{:,}",
                    "active_users": "{:,}",
                    "url_variants": "{:,.0f}"
                }),
                use_container_width=True,
                height=400
            )

            # Download buttons
            st.subheader("Download Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                excel_data = to_excel(per_person_df, totals_df)
                st.download_button(
                    label="Download Excel Report",
                    data=excel_data,
                    file_name=f"organization_{unit_id}_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with col2:
                csv_data = per_person_df.to_csv(index=False)
                st.download_button(
                    label="Download Per Person CSV",
                    data=csv_data,
                    file_name=f"organization_{unit_id}_per_person.csv",
                    mime="text/csv"
                )

            with col3:
                if not unmatched_df.empty:
                    unmatched_csv = unmatched_df.to_csv(index=False)
                    st.download_button(
                        label="Download Unmatched Paths",
                        data=unmatched_csv,
                        file_name=f"organization_{unit_id}_unmatched.csv",
                        mime="text/csv"
                    )

            # Show unmatched paths
            if not unmatched_df.empty:
                with st.expander(f"Unmatched GA4 Paths ({len(unmatched_df)} paths)"):
                    st.dataframe(unmatched_df, use_container_width=True)
            else:
                st.success("All GA4 paths were successfully matched!")

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")


if __name__ == "__main__":
    main()
