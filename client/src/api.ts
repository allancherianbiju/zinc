import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
  withCredentials: true,
});

// Add a request interceptor
api.interceptors.request.use((config) => {
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  if (user.email) {
    config.headers.Authorization = `Bearer ${user.email}`;
  }
  return config;
});

export default api;
