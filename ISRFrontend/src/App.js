import React from "react";
import Home from "./pages/Home";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import Search from "./components/Search";
import SearchPage from "./pages/SearchPage";
import Restaurant from "./pages/Restaurant";

function App() {
  return (
    <div className="app">
      <Router>
        <Switch>
          <Route path="/recommendation">
            <Restaurant/>
          </Route>
          <Route path="/search">
            <SearchPage />
          </Route>
          <Route path="/">
            <Home />
          </Route>
        </Switch>
      </Router>
    </div>
  );
}

export default App;

// https://developers.google.com/custom-search/v1/using_rest
// https://cse.google.com/cse/create/new
