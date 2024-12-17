import React, { useState, useEffect } from "react";
import "./SearchPage.css";
import useRestaurantSearch from "../useRestaurantSearch";
import { useStateValue } from "../StateProvider";
import { Link } from "react-router-dom";
import Search from "../components/Search";
import { useHistory } from "react-router-dom";
import { actionTypes } from "../reducer";
import logo from "../logo_one.png";
import ChatBox from "./chatbox";

function SearchPage() {
  const [{ term, zipCode }, dispatch] = useStateValue();

  // LIVE API CALL
  const { data } = useRestaurantSearch(term, zipCode);
  const history = useHistory();

  const [chatMessages, setChatMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (data) {
      setIsLoading(true);
      fetchChatData(data);
    }
  }, [data]);

  const fetchChatData = async (jsonData) => { 
    try {
      const url = `http://127.0.0.1:5000/chat-data`;
      const response = await fetch(url, {
        method: "POST", 
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ meta: jsonData }), 
      });
      if (!response.ok) {
        throw new Error("Failed to fetch chat data");
      }
      const responseData = await response.json(); 
      setChatMessages(responseData.messages);
    } catch (error) {
      console.error("Error fetching chat data:", error);
    } finally {
      setIsLoading(false);
    }
  };
  


  console.log(term)
  const recommend = (item) => {
    // Logic for recommending the item
    console.log("Item recommended:", item.gmap_id);
    dispatch({
      type: actionTypes.SET_RECOMMENDATION,
      gmap_id: item.gmap_id,
    });
    history.push("/recommendation/" + item.name);
  };

  return (
    <div className="searchPage">
      <div className="searchPage__header">
        <Link to="/">
          <img className="searchPage__logo" src={logo} alt="" />
        </Link>
        <div className="searchPage__headerBody">
          <Search hideButtons initialInput={term} initialZip={zipCode} />
        </div>
      </div>
      {term && (
        <div className="searchPage__results">
          <p className="searchPage__resultCount">
            {data === null ? `Loading results for "${term}" near ZIP code: ${zipCode}...` : `Results for "${term}" near ZIP code: ${zipCode}`}
          </p>
          {data?.items.map((item, index) => (
            <div className="searchPage__result" key={index}>
              <div
                className="searchPage__resultTitle"
                onClick={() => recommend(item)}
              >
                <h2>{item.name}</h2>
              </div>
              <p className="searchPage__resultSnippet">{item.description}</p>
            </div>
          ))}
          {data && isLoading && <ChatBox messages={[{"text": "Loading...", "type": "bot"}]}/>}
          {data && !isLoading && <ChatBox messages={chatMessages} />}
        </div>
      )}
    </div>
  );
}

export default SearchPage;
