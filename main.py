import math 
import json
import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler


def calculate_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    r = 3958.8
    distance = r * c
    return distance

def get_documents_within_25miles(df, latitude, longitude):
    locations_within_25km = df[df.apply(lambda row: calculate_distance(latitude, longitude, row['latitude'], row['longitude']), axis=1) < 25]
    return locations_within_25km

def distance_from_reference(location, zipcode):
    latitude, longitude = get_lat_lon(zipcode)
    return calculate_distance(latitude, longitude,
                              location["latitude"], location["longitude"])

def get_lat_lon(zipcode, country="US"):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'postalcode': zipcode,
        'country': country,
        'format': 'json'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data:
            # Taking the first result as the most relevant one
            latitude = data[0]['lat']
            longitude = data[0]['lon']
            return float(latitude), float(longitude)
        else:
            return None, None
    else:
        return None, None

# def rag(llm,meta_list, query):
#     keys_to_keep = ['name','description','avg_rating','num_of_reviews','MISC']
#     modified_list_of_dicts = [{k: v for k, v in d.items() if k in keys_to_keep} for d in meta_list[:5]]

#     result = json.dumps(modified_list_of_dicts)
#     prompt_template = """You are a restaurent recommender system that help users to find restaurents that match their preferences. Based on search query, suggest three restaurents, with a description, rating or MISC etc and the reason why the user will like it (reason should loosely be based on query).
#     Your response should be concise with one restaurent on one line. Use the following information for restaurants. 
#     "{context}"
#     Search Query: {query}
#     Your Response: """
#     # PROMPT = PromptTemplate(
#     # template=prompt_template, input_variables=["context", "query"])
#     llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
#     summary = llm_chain.run({"context": result, "query": query})
#     # print(summary)
#     return summary

# def ensemble(llm,retriever, query, zipcode):
#     docs = retriever.invoke(query)
#     meta_dict = {}

#     for doc in docs:
#         gmap_id = doc.metadata["gmap_id"]
#         if gmap_id not in meta_dict:
#             meta_dict[gmap_id] = doc.metadata

#     meta_list = list(meta_dict.values())
#     # summary = rag(llm,meta_list)
#     search_info = {}
#     # search_info['summary'] = summary
#     search_info['items'] = meta_list
#     search_info['query'] = query
#     return search_info

def retrieval_info(data, document_ids, bm25, query, zipcode):
    
    query_tokens = query.lower().split(" ")
    # Get document scores
    doc_scores = bm25.get_scores(query_tokens)
    data['doc_scores'] = doc_scores

    # latitude, longitude = get_lat_lon(zipcode)
    
    # docs = get_documents_within_25miles(data,latitude,longitude)
    
    # Normalize BM25 Scores
    scaler = MinMaxScaler()
    normalized_doc_scores = scaler.fit_transform(doc_scores.reshape(-1, 1)).flatten()

    # Normalize Avg Ratings (if they exist)
    avg_ratings = data['avg_rating'].fillna(0).to_numpy()  # Handle missing values
    normalized_avg_ratings = scaler.fit_transform(avg_ratings.reshape(-1, 1)).flatten()

    # Combine Scores (Weighted Sum)
    weight_bm25 = 0.7
    weight_rating = 0.3
    combined_scores = (
        weight_bm25 * normalized_doc_scores + weight_rating * normalized_avg_ratings
    )

    # Combine Document IDs, Scores, and Ratings
    doc_scores_with_ids = list(
        zip(document_ids, doc_scores, avg_ratings, combined_scores)
    )

    # Sort by Combined Score
    sorted_docs_with_ids = sorted(doc_scores_with_ids, key=lambda x: -x[3])

    # Extract the top 10 indices
    ind = [i for i, _, _, _ in sorted_docs_with_ids[:10]]

    # Retrieve corresponding rows from doc_df
    top_10_docs = data.loc[ind, :]

    # sorted_docs = docs.sort_values(by=['doc_scores', 'avg_rating'], ascending=[False, False])
    # print(sorted_docs)
    # top_10_docs = sorted_docs.head(10)
    top_10_docs.fillna(' ', inplace=True)
    top_10_docs_list = top_10_docs.to_dict(orient='records')
    search_info ={}
    search_info['items'] = top_10_docs_list
    print(search_info)
    return search_info




    