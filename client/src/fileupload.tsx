import axios from "axios";
import { Input } from "@nextui-org/react";
import { IconUpload } from "@tabler/icons-react";

function FileUpload() {
  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0]; // Use optional chaining
    if (!file) return; // Check if file is null
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("/upload", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      console.log(response.data);
    } catch (error) {
      console.error(error);
    }
  };
  return (
    <Input
      color="primary"
      startContent={<IconUpload className="w-4 h-4" />}
      type="file"
      onChange={handleFileUpload}
      accept=".csv"
    />
  );
}

export default FileUpload;
