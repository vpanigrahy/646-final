import { useState, useEffect } from "react";
import axios from "axios"; // Import Axios

function useRestaurantSearch(term, zipCode) {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Create request body
        const requestBody = {
          query: term,
          zipcode: zipCode
        };

        // Send POST request using Axios
        const response = await axios.post("http://127.0.0.1:5000/", requestBody, {
          headers: {
            "Content-Type": "application/json",
          }
        });
        // Set data to response data
        setData(response.data);
      } catch (error) {
        console.error("Error fetching data:", error);
        // Handle error as needed
      }
    };

    fetchData();
  }, [term, zipCode]); // Add zipCode to the dependency array since it's used in fetchData

  return { data };
}

export default useRestaurantSearch;
