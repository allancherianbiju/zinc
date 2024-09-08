import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Card, CardBody, Progress, Button } from "@nextui-org/react";
import Navbar from "../components/navbar";
import api from "../api";

const Processing = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Processing...");
  const [error, setError] = useState<string | null>(null);
  const { mapping, resolution_time_field } = location.state || {};

  useEffect(() => {
    const processData = async () => {
      try {
        setProgress(10);
        const response = await api.post("/process", {
          mapping,
          resolution_time_field,
        });
        setProgress(100);
        setStatus(response.data.message);
      } catch (error: any) {
        console.error("Error processing data:", error);
        setStatus("Error occurred while processing data");
        setError(error.response?.data?.detail || error.message);
        setProgress(100);
      }
    };

    processData();
  }, [mapping, resolution_time_field]);

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Navbar />
      <main className="flex-grow container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-4">Processing Data</h1>
        <Card className="mb-6">
          <CardBody>
            <Progress value={progress} className="mb-4" />
            <p>{status}</p>
            {error && (
              <p className="text-red-500 mt-2">Error details: {error}</p>
            )}
          </CardBody>
        </Card>
        {progress === 100 && !error && (
          <Button color="primary" onClick={() => navigate("/report")}>
            View Report
          </Button>
        )}
        {error && (
          <Button color="primary" onClick={() => navigate("/mapping")}>
            Back to Mapping
          </Button>
        )}
      </main>
    </div>
  );
};

export default Processing;
