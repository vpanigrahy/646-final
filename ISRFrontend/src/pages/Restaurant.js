import React from "react";
import { useStateValue } from "../StateProvider";
import useRecommendation from "../useRecommendation";
import "./Restaurant.css";

function Restaurant() {
  const [{ recommendFor }, dispatch] = useStateValue();
  const { recData } = useRecommendation(recommendFor);

  if (!recData) {
    return <div>Loading...</div>;
  }

  const restaurant1 = recData.restaurantInfo;
  const recommendations = recData.recommendations;
  const summary = recData.summary;

  return (
    <div className="restaurant-container">
      <div className="restaurant-wrapper">
        <div className="restaurant">
          <div className="restaurant-details">
            <h1>{restaurant1.name}</h1>
            <p className="category">{restaurant1.category.join(", ")}</p>
            <p className="address">{restaurant1.address}</p>
            <p className="description">{restaurant1.description}</p>
            <p className="rating">
              Rating: {restaurant1.avg_rating} ({restaurant1.num_of_reviews}{" "}
              reviews)
            </p>
            <p className="price">Price: {restaurant1.price}</p>
            <div className="hours">
              <p>Hours:</p>
              <ul>
                {restaurant1.hours.map((hour, index) => (
                  <li key={index}>
                    {hour[0]}: {hour[1]}
                  </li>
                ))}
              </ul>
            </div>
            <div className="misc">
              <p>Misc:</p>
              <ul>
                {Object.entries(restaurant1.MISC).map(([key, value]) => (
                  <li key={key}>
                    <strong>{key}: </strong>
                    {value.join(", ")}
                  </li>
                ))}
              </ul>
            </div>
            <a
              className="directions"
              href={restaurant1.url}
              target="_blank"
              rel="noopener noreferrer"
            >
              Get Directions
            </a>
          </div>
        </div>
      </div>
      <div className="summary-container">
        <h1>AI Generated Review Summarization </h1>
        <p className="summary">{summary}</p>
      </div>

      <div className="recommendations">
        <h2 className="headings-now">Restaurant Recommendations</h2>
        <div className="recommendation-items">
          {recommendations &&
            recommendations.map((recommendation, index) => (
              <div className="recommendation-item" key={index}>
                <h3>{recommendation.name}</h3>
                <p className="category">
                  {Array.isArray(recommendation.category) &&
                  recommendation.category.length > 1
                    ? recommendation.category.join(", ")
                    : recommendation.category}
                </p>
                <p className="address">{recommendation.address}</p>
                <p className="rating">
                  Rating: {recommendation.avg_rating} (
                  {recommendation.num_of_reviews} reviews)
                </p>
                <p className="price">Price: {recommendation.price}</p>
                {recommendation.url && (
                  <a
                    className="recommendation-url"
                    href={recommendation.url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    View Details
                  </a>
                )}
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

export default Restaurant;
