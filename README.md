# Pure_ Researcher_Profile_Analytics (GA4)

A simple, interactive web application for fetching related persons from an organizational unit in Pure from the Pure API and analyzing it with Google Analytics data. Get organization totals or breakdown per person.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

3. Open your browser (usually auto-opens at `http://localhost:8501`)

## How to Use

### Step 1: Configure Settings
- Enter your **Organization Unit ID** (e.g., `12345678`)
- Provide your **Pure API Key** (or set the `PURE_API_KEY` environment variable)
- Optionally adjust advanced settings in the sidebar

### Step 2: Fetch Persons from Pure
- Click "Fetch Persons from Pure API"
- Wait for the data to load (progress bar will show status)
- Review the fetched persons and their URL mappings

### Step 3: Upload GA4 Data
- Export your Google Analytics data as CSV with these columns:
  - `Page path and screen class`
  - `Views`
  - `Active users`
- Upload the CSV file
- The app will automatically join the data

### Step 4: View Results
- Organization totals (total views, active users, person count)
- Per-person breakdown (sortable and searchable table)
- Download results as Excel or CSV

## Features

- Real-time fetching from Pure API with progress tracking
- Automatic detection of CSV delimiter (`;` or `,`)
- Interactive data filtering and search
- Excel export with multiple sheets (totals + per-person breakdown)
- Unmatched GA4 paths report
- Session state management (data persists during session)

## Data Flow

1. Fetch persons from Pure organizational unit
2. Build URL mappings with prettyURL identifiers and UUIDs
3. Upload GA4 export data
4. Normalize and match URL paths
5. Aggregate analytics per person
6. Display and export results

## Environment Variables

- `PURE_API_KEY`: Your Pure API key (optional, can be entered in UI)

## File Outputs

The app generates downloadable files:
- `organization_{unit_id}_report.xlsx`: Complete report with totals and per-person data
- `organization_{unit_id}_per_person.csv`: Per-person analytics data
- `organization_{unit_id}_unmatched.csv`: GA4 paths that didn't match any person

## Notes

- The app uses session state to cache fetched data
- You can refresh GA4 analysis by uploading a new CSV without re-fetching Pure data
- URL matching is case-sensitive and normalized (removes trailing slashes, query params)
