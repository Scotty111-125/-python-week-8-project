import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# STREAMLIT APP SETUP

st.set_page_config(page_title="CORD-19 Mini Explorer", layout="wide")

st.title("CORD-19 Mini Data Explorer")
st.write("Exploring a smaller COVID-19 dataset for faster demo and debugging.")


# PART 1: Load Data

@st.cache_data
def load_data():
    df = pd.read_csv("mini_metadata.csv")  # small dataset
    return df

df = load_data()

st.header(" Part 1: Data Loading & Basic Exploration")
st.write("Dataset preview:")
st.dataframe(df.head())
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.write("Column data types:", df.dtypes)
st.write("Missing values per column:", df.isnull().sum())
st.write("Basic statistics:", df.describe(include='all'))


# PART 2: Data Cleaning

st.header("🧹 Part 2: Data Cleaning & Preparation")

df_cleaned = df.copy()
for col in ["title", "abstract", "journal"]:
    df_cleaned[col] = df_cleaned[col].fillna("")

df_cleaned["publish_time"] = pd.to_datetime(df_cleaned["publish_time"], errors="coerce")
df_cleaned["year"] = df_cleaned["publish_time"].dt.year
df_cleaned["abstract_word_count"] = df_cleaned["abstract"].apply(lambda x: len(str(x).split()))

st.write("Cleaned dataset preview:")
st.dataframe(df_cleaned)


# PART 3: Basic Analysis

st.header("Part 3: Analysis & Visualization")

# Publications per year
pubs_per_year = df_cleaned["year"].value_counts().sort_index()
fig, ax = plt.subplots()
pubs_per_year.plot(kind="bar", ax=ax)
ax.set_title("Publications per Year")
st.pyplot(fig)

# Top journals
top_journals = df_cleaned["journal"].value_counts().head(5)
fig, ax = plt.subplots()
top_journals.plot(kind="barh", ax=ax)
ax.set_title("Top Journals")
st.pyplot(fig)

# Word cloud
all_titles = " ".join(df_cleaned["title"].astype(str).tolist())
if all_titles.strip():
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_titles)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# Sources
source_counts = df_cleaned["source_x"].value_counts()
fig, ax = plt.subplots()
source_counts.plot(kind="bar", ax=ax)
ax.set_title("Distribution by Source")
st.pyplot(fig)


# PART 4: Interactivity

st.header(" Part 4: Interactive Exploration")

st.sidebar.header("Filters")
min_year, max_year = int(df_cleaned["year"].min()), int(df_cleaned["year"].max())
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

df_filtered = df_cleaned[(df_cleaned["year"] >= year_range[0]) & (df_cleaned["year"] <= year_range[1])]

journals = ["All"] + df_cleaned["journal"].unique().tolist()
selected_journal = st.sidebar.selectbox("Select Journal", journals)
if selected_journal != "All":
    df_filtered = df_filtered[df_filtered["journal"] == selected_journal]

keyword = st.sidebar.text_input("Search Keyword").lower().strip()
if keyword:
    df_filtered = df_filtered[
        df_filtered["title"].str.lower().str.contains(keyword) |
        df_filtered["abstract"].str.lower().str.contains(keyword)
    ]

st.write(f"Showing {df_filtered.shape[0]} papers after filters:")
st.dataframe(df_filtered)

