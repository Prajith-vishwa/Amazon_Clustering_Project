import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(
    page_title="AMAZON MUSIC CLUSTERING",
    layout="wide"
)

# ----------------------------------------------------
# 🌟 DARK BACKGROUND COLOR CUSTOMIZATION
# ----------------------------------------------------

# Inject CSS to change the main page and sidebar background colors to a dark theme
st.markdown(
    """
    <style>
    /* Main Content Area: Dark Slate Gray */
    .stApp {
        background-color: #2F4F4F; 
        color: white; /* Ensure text is readable on the dark background */
    }
    /* Sidebar Area: Slightly lighter Gray for contrast */
    .css-1d391kg {
        background-color: #4F4F4F; 
        color: white; /* Ensure text in sidebar is readable */
    }
    /* Fix for main content text color on dark background */
    section.main p {
        color: white;
    }
    /* Fix for subheaders/titles (may require manual adjustment if Streamlit doesn't handle them automatically) */
    .stHeading, .stTitle, h1, h2, h3, h4, .css-10trblm {
        color: white !important; 
    }
    /* Adjust text in metrics for better visibility */
    .stMetricLabel {
        color: lightgray !important;
    }
    .stMetricValue {
        color: #ADFF2F !important; /* Brighter accent for values */
    }
    /* Adjust text in dataframes for better visibility */
    .stDataFrame {
        color: black; /* Dataframes often look better with a light background and dark text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------
# 🎨 PLOT CUSTOMIZATION AND STYLING
# ----------------------------------------------------

# 1. Set global Seaborn theme for a different template (switching to 'darkgrid' complements the dark app theme)
sns.set_theme(style="darkgrid")

# 2. Define a consistent color palette for clusters (e.g., 'Paired' or a vibrant dark-friendly palette)
CUSTOM_PALETTE = "viridis" # 'viridis' works well on dark backgrounds
sns.set_palette(CUSTOM_PALETTE)

# 3. Define a different color map for the Heatmap (e.g., 'magma' or 'rocket' for dark themes)
HEATMAP_CMAP = "magma"

# Customize Matplotlib parameters for cleaner text/titles
plt.rcParams['figure.dpi'] = 100 
plt.rcParams['axes.titlesize'] = 16 
plt.rcParams['axes.labelsize'] = 12 
plt.rcParams['figure.autolayout'] = True 
plt.rcParams['figure.facecolor'] = '#363636' # Dark background for the plot area itself
plt.rcParams['axes.facecolor'] = '#363636'  # Dark background for the axes
plt.rcParams['text.color'] = 'white'      # White text for titles/labels
plt.rcParams['xtick.color'] = 'white'     # White tick marks
plt.rcParams['ytick.color'] = 'white'     # White tick marks
plt.rcParams['axes.labelcolor'] = 'white' # White axis labels


# ----------------------------------------------------
# DATA LOADING
# ----------------------------------------------------

# Load preprocessed data from pickle
@st.cache_data
def load_preprocessed(path="cleaned_data.pkl"):
    """Loads preprocessed data objects from the pickle file."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error(f"Error: The file '{path}' was not found. Please ensure it is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Load all objects from pickle
data = load_preprocessed()

df_reference = data["df_reference"]              
df_standard_scaled = data["df_standard_scaled"]  
df_pca = data["df_pca"]                          
kmeans_labels = data["kmeans_labels"]            
cluster_profile = data["cluster_profile"]        
feature_columns = data["feature_columns"]        
sil_score = data.get("sil_score", None)          
db_index = data.get("db_index", None)            

# Ensure cluster column exists
df_reference['Cluster'] = kmeans_labels

# Sidebar - Tab selection
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["MUSIC-OVERVIEW", "MUSIC-METRICS", "MUSIC-VISUALIZATION", "MUSIC-ANALYSIS"])

# ----------------------------------------------------
# PAGE CONTENT
# ----------------------------------------------------

# Overview
if page == "MUSIC-OVERVIEW":
    st.title("AMAZON MUSIC CLUSTERING")
    st.markdown("""
    This project groups similar Amazon Music songs based on their audio features using **K-Means clustering**.
    Explore clusters, visualize patterns, and understand musical characteristics.
    """)
    
    st.subheader("DATASET SUMMARY")
    st.write(f"Total Songs: {df_reference.shape[0]}")
    st.write(f"Audio Features Used: {len(feature_columns)}")
    st.write(f"Number of Clusters: {df_reference['Cluster'].nunique()}")
    
    st.subheader("TOP 5 SONGS")
    st.dataframe(df_reference[['name_song', 'name_artists', 'genres', 'Cluster']].head(5))
    
    st.subheader("CLUSTER DISTRIBUTION")
    cluster_counts = df_reference['Cluster'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8,5))
    # Customized palette
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette=CUSTOM_PALETTE, ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Songs")
    ax.set_title("Cluster Size Distribution")
    st.pyplot(fig)

# Metrics
elif page == "MUSIC-METRICS":
    st.title("EVALUATION METRICS-CLUSTER")
    
    # Show precomputed scores
    st.metric("SILHOUETTE SCORE", round(sil_score, 4) if sil_score is not None else "Not available")
    st.metric("DAVIES-BOULDIN INDEX", round(db_index, 4) if db_index is not None else "Not available")
    
    st.subheader("CLUSTER WISE MEAN")
    st.dataframe(cluster_profile)

# Visualization
elif page == "MUSIC-VISUALIZATION":
    st.title("CLUSTER VISUALIZATIONS")
    
    # PCA 2D Scatter Plot
    st.subheader("PCA 2D Scatter Plot")
    pca_df = pd.DataFrame(df_pca[:, :2], columns=['PC1', 'PC2'])
    pca_df['Cluster'] = kmeans_labels
    
    fig, ax = plt.subplots(figsize=(10,7))
    # Customized palette
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette=CUSTOM_PALETTE, s=50, ax=ax)
    ax.set_title("PCA VISUALIZATIONS OF CLUSTERS")
    ax.grid(True)
    st.pyplot(fig)
    
    # Heatmap of Cluster Feature Means
    st.subheader("HEATMAP OF CLUSTERS")
    fig, ax = plt.subplots(figsize=(12,6))
    # Customized color map (cmap)
    sns.heatmap(cluster_profile, annot=True, fmt=".2f", cmap=HEATMAP_CMAP, ax=ax)
    ax.set_title("CLUSTER-WISE COMPARISON")
    st.pyplot(fig)
    
    # Key Feature Averages per Cluster
    st.subheader("KEY FEATURES AVERAGE OF CLUSTERS")
    features_to_compare = ['energy', 'acousticness', 'valence', 'instrumentalness', 'speechiness']
    fig, ax = plt.subplots(figsize=(10,6))
    # The plot method uses the global Seaborn palette set earlier
    cluster_profile[features_to_compare].plot(kind='bar', ax=ax)
    ax.set_ylabel("Scaled Feature Value")
    ax.set_xlabel("Cluster")
    ax.set_title("Key Features by Cluster")
    ax.grid(axis='y', linestyle='--') # Custom grid line style
    plt.xticks(rotation=0) # Ensure cluster labels are horizontal
    st.pyplot(fig)

# Insights & Export
elif page == "MUSIC-ANALYSIS":
    st.title("ANALYSIS")
    
    st.subheader("TOP SONGS PER CLUSTER")
    cluster_select = st.selectbox("Select Cluster", sorted(df_reference['Cluster'].unique()))
    top_n = st.slider("Number of Top Songs to Display", 5, 20, 5)
    
    top_songs = df_reference[df_reference['Cluster'] == cluster_select][['name_song', 'name_artists', 'genres']].head(top_n)
    st.dataframe(top_songs)
    
    st.subheader("CLUSTER INTERPRETATION")
    st.markdown("""
    - **Cluster 0**: High danceability, high energy → Party tracks  
    - **Cluster 1**: Low energy, high acousticness → Chill acoustic  
    - **Cluster 2**: Medium energy & valence → Balanced mood  
    - **Cluster 3**: Instrumental-heavy tracks → Relaxed/Focus tracks
    """)
    
    st.subheader("DATASET DOWNLOAD")
    csv = df_reference.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "AmazonMusic_Clustered.csv", "text/csv")