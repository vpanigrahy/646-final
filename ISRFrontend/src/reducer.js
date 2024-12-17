export const initialState = {
  term: null,
};

export const actionTypes = {
  SET_SEARCH_TERM: "SET_SEARCH_TERM",
  SET_RECOMMENDATION: "SET_RECOMMENDATION",
};

const reducer = (state, action) => {
  console.log(action);
  switch (action.type) {
    case actionTypes.SET_SEARCH_TERM:
      return {
        ...state,
        term: action.term,
        zipCode : action.zipCode
      };
    case actionTypes.SET_RECOMMENDATION:
      return {
        ...state,
        recommendFor: action.gmap_id,
      };

    default:
      return state;
  }
};

export default reducer;
