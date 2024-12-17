import { useState, useEffect } from "react";
import axios from "axios";

import recommendationResponse from "./recommendationResponse";
function useRecommendation(recommendFor) {
  const [recData, setRecData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Create request body
        const requestBody = {
          gmap_id: recommendFor,
        };

        // Send POST request
        const response = await axios.post("http://127.0.0.1:5000/search", requestBody, {
          headers: {
            "Content-Type": "application/json",
          }
        });
        setRecData(response.data);
        
      } catch (error) {
        console.error("Error fetching data using sample response:", error);
        setRecData(recommendationResponse);
      }
    };

    fetchData();
  }, [recommendFor]);

  return { recData };
}

export default useRecommendation;
