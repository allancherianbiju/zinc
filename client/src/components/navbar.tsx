import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Navbar,
  NavbarBrand,
  NavbarContent,
  NavbarItem,
  Dropdown,
  DropdownTrigger,
  DropdownMenu,
  DropdownItem,
  Avatar,
  Button,
  Tooltip,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Input,
  useDisclosure,
  ScrollShadow,
} from "@nextui-org/react";
import { useTheme } from "next-themes";
import {
  IconSun,
  IconMoon,
  IconLogout,
  IconSettings,
  IconSparkles,
  IconSend,
  IconBarrierBlock,
  IconMessage,
  IconMessagePlus,
} from "@tabler/icons-react";
import { Link } from "react-router-dom";

export default function AppNavbar() {
  const { theme, setTheme } = useTheme();
  const user = JSON.parse(localStorage.getItem("user") || "null");
  const [engagement, setEngagement] = useState<string>("");
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [messages, setMessages] = useState<
    Array<{ role: string; content: string }>
  >([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const handleStorageChange = () => {
      const currentEngagement = localStorage.getItem("currentEngagement");
      if (currentEngagement) {
        setEngagement(JSON.parse(currentEngagement).name);
      }
    };

    // Initial check
    handleStorageChange();

    // Listen for changes
    window.addEventListener("storage", handleStorageChange);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
    };
  }, []);

  useEffect(() => {
    // Load chat history from localStorage
    const savedMessages = localStorage.getItem("zincAIChat");
    if (savedMessages) {
      setMessages(JSON.parse(savedMessages));
    }
  }, []);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, { role: "user", content: input }];
    setMessages(newMessages);
    localStorage.setItem("zincAIChat", JSON.stringify(newMessages));
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: input,
        }),
      });

      const data = await response.json();
      const updatedMessages = [
        ...newMessages,
        { role: "assistant", content: data.response },
      ];
      setMessages(updatedMessages);
      localStorage.setItem("zincAIChat", JSON.stringify(updatedMessages));
    } catch (error) {
      console.error("Error sending message:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignOut = () => {
    localStorage.removeItem("user");
    localStorage.removeItem("currentEngagement");
    window.location.href = "/";
  };

  return (
    <>
      <Navbar isBordered maxWidth="full">
        <div className="container mx-auto px-4 flex justify-between items-center w-full">
          <NavbarContent justify="start">
            <NavbarBrand>
              <Link to="/" className="font-bold text-inherit no-underline">
                ZINC {engagement && `/ ${engagement}`}
              </Link>
            </NavbarBrand>
          </NavbarContent>

          <NavbarContent justify="end">
            <div className="flex items-center gap-2">
              {/* Zinc Intelligence */}
              <Tooltip showArrow={true} content="Zinc Intelligence">
                <Button isIconOnly variant="light" onPress={onOpen}>
                  <IconSparkles size={20} />
                </Button>
              </Tooltip>
              {/* <IconSparkles size={20} />
              <p className="text-sm font-semibold">Zinc AI</p> */}
            </div>
            {user && (
              <NavbarItem>
                <Dropdown placement="bottom-end">
                  <DropdownTrigger>
                    <Avatar
                      isBordered
                      className="transition-transform"
                      name={user.name}
                      size="sm"
                      src={user.picture}
                    />
                  </DropdownTrigger>
                  <DropdownMenu aria-label="Profile Actions" variant="flat">
                    <DropdownItem key="profile" className="h-14 gap-2">
                      <p className="font-semibold">Signed in as</p>
                      <p className="font-semibold">{user.email}</p>
                    </DropdownItem>
                    <DropdownItem
                      key="theme"
                      startContent={
                        theme === "light" ? <IconMoon /> : <IconSun />
                      }
                      onPress={() =>
                        setTheme(theme === "light" ? "dark" : "light")
                      }
                    >
                      Switch to {theme === "light" ? "Dark" : "Light"} mode
                    </DropdownItem>
                    <DropdownItem
                      key="settings"
                      startContent={<IconSettings />}
                      onPress={() => (window.location.href = "/settings")}
                    >
                      Settings
                    </DropdownItem>
                    <DropdownItem
                      key="logout"
                      color="danger"
                      startContent={<IconLogout />}
                      onPress={handleSignOut}
                    >
                      Log Out
                    </DropdownItem>
                  </DropdownMenu>
                </Dropdown>
              </NavbarItem>
            )}
          </NavbarContent>
        </div>
      </Navbar>

      <Modal
        isOpen={isOpen}
        onClose={onClose}
        size="3xl"
        scrollBehavior="inside"
      >
        <ModalContent>
          <ModalHeader className="flex flex-col gap-1">
            <span className="flex items-center gap-2">
              <IconSparkles size={20} /> Zinc Intelligence
              <Tooltip
                showArrow={true}
                content="Work in Progress"
                placement="bottom"
              >
                <Button
                  isIconOnly
                  variant="light"
                  color="warning"
                  onPress={onOpen}
                >
                  <IconBarrierBlock size={20} />
                </Button>
              </Tooltip>
            </span>
          </ModalHeader>
          <ModalBody>
            <ScrollShadow size={20} hideScrollBar>
              <div className="flex flex-col gap-4">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`flex items-start gap-2 my-2 ${
                      message.role === "user" ? "flex-row-reverse" : "flex-row"
                    }`}
                  >
                    {message.role === "user" ? (
                      <Avatar size="sm" src={user?.picture} name={user?.name} />
                    ) : (
                      <Avatar size="sm" name="Zi" color="secondary" />
                    )}
                    <div
                      className={`p-3 rounded-lg max-w-[80%] ${
                        message.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-default-100"
                      }`}
                    >
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex items-start gap-2">
                    <Avatar size="sm" name="Zi" color="secondary" />
                    <div className="p-3 rounded-lg bg-default-100">
                      Thinking...
                    </div>
                  </div>
                )}
              </div>
            </ScrollShadow>
          </ModalBody>
          <ModalFooter>
            <div className="flex w-full gap-2 mb-2">
              <Button
                isIconOnly
                color="default"
                variant="light"
                onPress={() => {
                  setMessages([]);
                  setInput("");
                }}
              >
                <IconMessagePlus size={20} />
              </Button>
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && sendMessage()}
                placeholder="Chat with Zinc Intelligence..."
                className="flex-grow"
                // startContent={<IconMessage size={20} />}
              />
              <Button
                isIconOnly
                color="primary"
                onPress={sendMessage}
                isLoading={isLoading}
              >
                <IconSend size={20} />
              </Button>
            </div>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  );
}
