import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Card,
  CardBody,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Button,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
  Pagination,
  Divider,
} from "@nextui-org/react";
import Navbar from "../components/navbar";
import api from "../api";
import ReactMarkdown from "react-markdown";
import { IconCopy, IconCheck } from "@tabler/icons-react";

const Report = () => {
  const navigate = useNavigate();
  const [reportData, setReportData] = useState<any>(null);
  const [selectedRow, setSelectedRow] = useState<any>(null);
  const [sop, setSop] = useState<string>("");
  const [page, setPage] = useState(1);
  const rowsPerPage = 10;
  const [sortField, setSortField] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");
  const [isGeneratingSOP, setIsGeneratingSOP] = useState(false);
  const [isCopied, setIsCopied] = useState(false);

  const { isOpen, onOpen, onClose } = useDisclosure();

  useEffect(() => {
    const fetchReportData = async () => {
      try {
        const user = JSON.parse(localStorage.getItem("user") || "{}");
        const response = await api.get(`/report/${user.id}`);
        setReportData(response.data);
      } catch (error) {
        console.error("Error fetching report data:", error);
      }
    };

    fetchReportData();
  }, []);

  const handleOpenModal = (row: any) => {
    setSelectedRow(row);
    setSop("");
    onOpen();
  };

  const handleGenerateSOP = async () => {
    if (!selectedRow) return;
    setIsGeneratingSOP(true);
    try {
      const response = await api.post("/generate_sop", {
        issue_description: selectedRow.issue_description,
        resolution_notes: selectedRow.resolution_notes,
      });
      setSop(response.data.sop);
    } catch (error) {
      console.error("Error generating SOP:", error);
    } finally {
      setIsGeneratingSOP(false);
    }
  };

  const handleCopySOP = () => {
    navigator.clipboard.writeText(sop);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  if (!reportData) {
    return <div>Loading...</div>;
  }

  const sortedData = [...reportData.table_data].sort((a, b) => {
    if (!sortField) return 0;
    const aValue = a[sortField];
    const bValue = b[sortField];
    if (aValue < bValue) return sortDirection === "asc" ? -1 : 1;
    if (aValue > bValue) return sortDirection === "asc" ? 1 : -1;
    return 0;
  });

  const paginatedData = sortedData.slice(
    (page - 1) * rowsPerPage,
    page * rowsPerPage
  );

  const handleCloseModal = () => {
    if (!isGeneratingSOP) {
      onClose();
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <Navbar />
      <main className="flex-grow container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-4">Incident Report</h1>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <Card>
            <CardBody>
              <h2 className="text-xl font-semibold mb-2">
                Most Complex Incident Type
              </h2>
              <p>Category: {reportData.cards.most_complex.category}</p>
              <p>Subcategory: {reportData.cards.most_complex.subcategory}</p>
              <p>Symptom: {reportData.cards.most_complex.u_symptom}</p>
              <p>Count: {reportData.cards.most_complex.count}</p>
            </CardBody>
          </Card>
          <Card>
            <CardBody>
              <h2 className="text-xl font-semibold mb-2">Date Range</h2>
              <p>
                From:{" "}
                {new Date(
                  reportData.cards.date_range.min_date
                ).toLocaleDateString()}
              </p>
              <p>
                To:{" "}
                {new Date(
                  reportData.cards.date_range.max_date
                ).toLocaleDateString()}
              </p>
            </CardBody>
          </Card>
          <Card>
            <CardBody>
              <h2 className="text-xl font-semibold mb-2">
                Average Resolution Time
              </h2>
              <p>
                {reportData.cards.avg_resolution_time.avg_resolution_time.toFixed(
                  2
                )}{" "}
                minutes
              </p>
            </CardBody>
          </Card>
        </div>

        <Table aria-label="Incident data table">
          <TableHeader>
            <TableColumn>Category</TableColumn>
            <TableColumn>Subcategory</TableColumn>
            <TableColumn>Symptom</TableColumn>
            <TableColumn
              onClick={() => handleSort("incident_count")}
              className="cursor-pointer"
            >
              Incident Count{" "}
              {sortField === "incident_count" &&
                (sortDirection === "asc" ? "▲" : "▼")}
            </TableColumn>
            <TableColumn
              onClick={() => handleSort("avg_resolution_time")}
              className="cursor-pointer"
            >
              Avg. Resolution Time (min){" "}
              {sortField === "avg_resolution_time" &&
                (sortDirection === "asc" ? "▲" : "▼")}
            </TableColumn>
            <TableColumn>Actions</TableColumn>
          </TableHeader>
          <TableBody>
            {paginatedData.map((row: any, index: number) => (
              <TableRow key={index}>
                <TableCell>{row.category}</TableCell>
                <TableCell>{row.subcategory}</TableCell>
                <TableCell>{row.u_symptom}</TableCell>
                <TableCell>{row.incident_count}</TableCell>
                <TableCell>{row.avg_resolution_time.toFixed(2)}</TableCell>
                <TableCell>
                  <Button
                    size="sm"
                    onClick={() => handleOpenModal(row)}
                    disabled={isGeneratingSOP}
                  >
                    View Details
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        <div className="flex justify-center mt-4">
          <Pagination
            total={Math.ceil(reportData.table_data.length / rowsPerPage)}
            page={page}
            onChange={setPage}
          />
        </div>

        <Modal
          isOpen={isOpen}
          onClose={handleCloseModal}
          size="4xl"
          scrollBehavior="inside"
          isDismissable={!isGeneratingSOP}
          isKeyboardDismissDisabled={isGeneratingSOP}
        >
          <ModalContent>
            <ModalHeader className="flex flex-col gap-1">
              Incident Details
            </ModalHeader>
            <ModalBody>
              {selectedRow && (
                <>
                  <h3 className="text-lg font-semibold">Issue Description</h3>
                  <p>{selectedRow.issue_description}</p>
                  <Divider className="my-4" />
                  <h3 className="text-lg font-semibold">Resolution Notes</h3>
                  <p>{selectedRow.resolution_notes}</p>
                  {sop && (
                    <>
                      <Divider className="my-4" />
                      <h3 className="text-lg font-semibold">Generated SOP</h3>
                      <div className="markdown-body">
                        <ReactMarkdown>{sop}</ReactMarkdown>
                      </div>
                    </>
                  )}
                </>
              )}
            </ModalBody>
            <ModalFooter>
              {!sop ? (
                <Button
                  color="primary"
                  onClick={handleGenerateSOP}
                  isLoading={isGeneratingSOP}
                >
                  {isGeneratingSOP ? "Generating..." : "Generate SOP"}
                </Button>
              ) : (
                <Button
                  color="primary"
                  onClick={handleCopySOP}
                  startContent={
                    isCopied ? <IconCheck size={18} /> : <IconCopy size={18} />
                  }
                >
                  {isCopied ? "Copied!" : "Copy SOP"}
                </Button>
              )}
              {!isGeneratingSOP && (
                <Button color="secondary" onClick={onClose}>
                  Close
                </Button>
              )}
            </ModalFooter>
          </ModalContent>
        </Modal>
      </main>
    </div>
  );
};

export default Report;
