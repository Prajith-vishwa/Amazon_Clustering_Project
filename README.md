# Amazon_Clustering_Project
This project creates an interactive dashboard for grouping similar Amazon Music songs using K-Means clustering based on their audio features.

**Project Overview** 
Objective: Cluster songs into groups based on characteristics like danceability, energy, and acousticness.

Tech Stack: Python, Streamlit, Pandas, Scikit-learn, Matplotlib, Seaborn, and Pickle.

Key Features: Includes PCA for dimensionality reduction, K-Means clustering, evaluation metrics (Silhouette Score & Davies-Bouldin Index), and an interactive dashboard.

**Dataset Details**
Source File: single_genre_artists.csv.

Features Used for Clustering: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, and duration_ms.

**Project Structure**
The main files in the project directory (Amazon Music Clustering/) are:

app.py: The primary Streamlit dashboard script.

cleaned_data.pkl: Pickle file containing the preprocessed data.

AmazonMusic_Clustered.csv: The final dataset exported with cluster labels.

requirements.txt: Lists all Python dependencies.

**Clustering Process (Pipeline)**
The process involves a standard ML pipeline:

Data Cleaning: Remove unnecessary ID and date columns.

Scaling: Use StandardScaler to normalize features.

Dimensionality Reduction: Use PCA to reduce feature space, retaining 95% variance.

Clustering: Apply K-Means (optimal clusters determined via elbow method).

Evaluation: Measure quality with Silhouette Score and Davies-Bouldin Index.

**Example Cluster Interpretation**
The clusters are interpreted based on the average feature values within the group:

Cluster 0: High danceability & energy → Party tracks.

Cluster 1: Low energy & high acousticness → Chill acoustic tracks.

Cluster 3: Instrumental-heavy → Relaxed/Focus tracks.
