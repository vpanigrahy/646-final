import React, { useState } from "react";
import "./Search.css";
import Button from "@material-ui/core/Button";
import { useHistory } from "react-router-dom";
import { useStateValue } from "../StateProvider";
import { actionTypes } from "../reducer";

function Search({ hideButtons = false, initialInput = "", initialZip = "" }) {
  const [{ term, zipCode }, dispatch] = useStateValue();
  const [input, setInput] = useState(initialInput || term);
  const [zipInput, setZipInput] = useState(initialZip || zipCode);
  const history = useHistory();

  const search = (e) => {
    e.preventDefault();

    if (!input.trim() || !zipInput.trim()) {
      alert("Please enter both search term and ZIP code.");
      return;
    }

    console.log("button clicked", input, zipInput);

    dispatch({
      type: actionTypes.SET_SEARCH_TERM,
      term: input,
      zipCode: zipInput
    });

    history.push("/search");
  };

  const handleZipChange = (e) => {
    setZipInput(e.target.value);
  };

  return (
    <form className="search">
      <div className="search__input">
        <input 
          placeholder="Explore Restaurants"
          value={input} 
          onChange={(e) => setInput(e.target.value)} 
        />
      </div>
      <div className="search__input">
        <input
          placeholder="Enter ZIP Code"
          value={zipInput}
          onChange={handleZipChange}
        />
      </div>
      {!hideButtons ? (
        <div className="search__buttons">
          <Button onClick={search} type="submit" variant="outlined">
            Restaurant Search
          </Button>
        </div>
      ) : (
        <div className="search__buttons">
          <Button
            className="search__buttonsHidden"
            onClick={search}
            type="submit"
            variant="outlined"
          >
            Restaurant Search
          </Button>
        </div>
      )}
    </form>
  );
}
export default Search;
