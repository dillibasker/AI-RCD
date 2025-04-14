// src/App.jsx
import { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [userId, setUserId] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [error, setError] = useState('');

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!userId) {
      setError('Please enter a user ID');
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/recommend', {
        user_id: userId,
        search_query: searchQuery,
      });
      setRecommendations(response.data.recommendations);
      setError('');
    } catch (err) {
      setError('Failed to fetch recommendations');
      console.error(err);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Recommendation System</h1>
        <form onSubmit={handleSearch}>
          <div className="form-group">
            <label>User ID:</label>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="Enter user ID (e.g., user_1)"
            />
          </div>
          <div className="form-group">
            <label>Search:</label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search tags (e.g., tag_1)"
            />
          </div>
          <button type="submit">Get Recommendations</button>
        </form>
        {error && <p className="error">{error}</p>}
        <div className="recommendations">
          <h2>Recommendations</h2>
          <ul>
            {recommendations.length > 0 ? (
              recommendations.map((item, index) => (
                <li key={index}>{item}</li>
              ))
            ) : (
              <p>No recommendations yet</p>
            )}
          </ul>
        </div>
      </header>
    </div>
  );
}

export default App;