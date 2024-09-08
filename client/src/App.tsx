import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Home from "./home"; // Add this line
import Preview from "./preview";
import ColumnMapping from "./pages/ColumnMapping";
import Processing from "./pages/Processing";
import Report from "./pages/Report";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/preview" element={<Preview />} />
        <Route path="/mapping" element={<ColumnMapping />} />
        <Route path="/processing" element={<Processing />} />
        <Route path="/report" element={<Report />} />
      </Routes>
    </Router>
  );
}

export default App;
