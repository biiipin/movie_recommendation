import streamlit as st
import pickle
import requests
from sklearn.neighbors import NearestNeighbors
import difflib

st.markdown(
    """
    <style>
    .main {
        background-image: url('https://images.unsplash.com/photo-1517602302552-471fe67acf66?auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-position: center;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


movies = pickle.load(open("movies_data.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))

st.title("🎬 Movie Recommender 🎬")


movie_names = movies['title'].values
selected_movie = st.selectbox("Type or select a movie", movie_names)


def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=bb8c8e12742c72ae502a3863ccb5402a&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    else:
        return "https://via.placeholder.com/500x750?text=No+Image"


def fetch_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=bb8c8e12742c72ae502a3863ccb5402a&language=en-US"
    data = requests.get(url).json()
    overview = data.get('overview', 'No description available.')
    release_date = data.get('release_date', 'Unknown')
    rating = data.get('vote_average', 0)
    return overview, release_date, rating

def fetch_trailer(movie_id):
    url=f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key=bb8c8e12742c72ae502a3863ccb5402a&language=en-US"
    data=requests.get(url).json()
    movie_trailer=data.get('results',[])
    for video in movie_trailer:
        if video['type']=='Trailer' and video['site']=='YouTube':
            return f"https://www.youtube.com/watch?v={video['key']}"

    return "No trailer"

nbrs = NearestNeighbors(n_neighbors=6, metric='cosine').fit(tfidf_matrix)


def recommend(title):
    matches = difflib.get_close_matches(title.lower(), movies['title'].str.lower(), n=1, cutoff=0.6)
    if not matches:
        return [], [], [], [], [],[]
    idx = movies[movies['title'].str.lower() == matches[0]].index[0]
    distances, indices = nbrs.kneighbors(tfidf_matrix[idx])

    rec_titles = []
    rec_posters = []
    rec_overviews = []
    rec_dates = []
    rec_ratings = []
    rec_trailers=[]

    for i in indices[0][1:]:
        movie_id = movies.iloc[i].id
        rec_titles.append(movies.iloc[i].title)
        rec_posters.append(fetch_poster(movie_id))
        overview, date, rating = fetch_details(movie_id)
        rec_overviews.append(overview)
        rec_dates.append(date)
        rec_ratings.append(rating)
        trailer_link=fetch_trailer(movie_id)
        rec_trailers.append(trailer_link)

    return rec_titles, rec_posters, rec_overviews, rec_dates, rec_ratings,rec_trailers

if st.button("Show Recommendations"):
    with st.spinner("Finding movies for you... 🎥"):
        titles, posters, overviews, dates, ratings,rec_trailers= recommend(selected_movie)
        if titles:
            cols = st.columns(5)
            for i in range(len(titles)):
                with cols[i]:
                    st.image(posters[i])
                    st.markdown(f"### {titles[i]}")
                    st.markdown(f"**Release:** {dates[i]}")
                    st.markdown(f"**Rating:** ({ratings[i]})")
                    st.write(overviews[i][:200] + "...")
                    trailer_url = rec_trailers[i]
                    if trailer_url != "No trailer":
                        st.video(trailer_url)
                    else:
                        st.write("No trailer available")

        else:
            st.error("No matching movies found. Try another title.")
