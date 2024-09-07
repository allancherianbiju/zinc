import React from "react";
import {
  Button,
  Input,
  Accordion,
  AccordionItem,
  Card,
  CardBody,
  Tabs,
  Tab,
  ScrollShadow,
} from "@nextui-org/react";
import {
  IconBolt,
  IconBook,
  IconMessage,
  IconSend,
  IconUser,
  IconUsers,
  IconLock,
} from "@tabler/icons-react";

export default function GG() {
  const [messages, setMessages] = React.useState([
    {
      role: "assistant",
      content:
        "Hello! How can I help you learn about AI today? Here are some questions you can ask:",
    },
    { role: "assistant", content: "• What is Generative AI?" },
    { role: "assistant", content: "• How does machine learning work?" },
    {
      role: "assistant",
      content: "• What are some applications of AI in business?",
    },
  ]);
  const [inputMessage, setInputMessage] = React.useState("");

  const handleSendMessage = () => {
    if (inputMessage.trim() !== "") {
      setMessages([...messages, { role: "user", content: inputMessage }]);
      setInputMessage("");
      // Here you would typically send the message to your AI backend and get a response
      // For this example, we'll just simulate a response after a short delay
      setTimeout(() => {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content:
              "Thank you for your question. To access more detailed responses and advanced chat features, please upgrade to our premium plan.",
          },
        ]);
      }, 1000);
    }
  };

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <Card className="w-64 h-full hidden md:flex flex-col rounded-none">
        <CardBody className="p-4">
          <h2 className="text-2xl font-bold mb-4">GenAI Gurukul</h2>
          <ScrollShadow className="flex-grow">
            <Accordion>
              <AccordionItem
                key="learning-resources"
                aria-label="Learning Resources"
                title="Learning Resources"
              >
                <nav>
                  <ul className="space-y-2">
                    <li>
                      <a href="#" className="text-primary hover:underline">
                        Overview
                      </a>
                    </li>
                    <li>
                      <a href="#" className="text-primary hover:underline">
                        Architecture
                      </a>
                    </li>
                    <li>
                      <a href="#" className="text-primary hover:underline">
                        Business Benefits
                      </a>
                    </li>
                    <li>
                      <a href="#" className="text-primary hover:underline">
                        Examples
                      </a>
                    </li>
                    <li>
                      <a href="#" className="text-primary hover:underline">
                        Opportunities
                      </a>
                    </li>
                  </ul>
                </nav>
              </AccordionItem>
              <AccordionItem
                key="learning-journey"
                aria-label="Learning Journey"
                title="Learning Journey"
              >
                <nav>
                  <ul className="space-y-2">
                    <li>
                      <a href="#" className="text-primary hover:underline">
                        My Progress
                      </a>
                    </li>
                    <li>
                      <a href="#" className="text-primary hover:underline">
                        Certificates
                      </a>
                    </li>
                    <li>
                      <a href="#" className="text-primary hover:underline">
                        Skill Assessment
                      </a>
                    </li>
                  </ul>
                </nav>
              </AccordionItem>
              <AccordionItem
                key="mentorship"
                aria-label="Mentorship"
                title="Mentorship"
              >
                <div className="space-y-2">
                  <Button
                    variant="bordered"
                    className="w-full justify-start"
                    startContent={<IconUser />}
                  >
                    Individual Mentorship
                  </Button>
                  <Button
                    variant="bordered"
                    className="w-full justify-start"
                    startContent={<IconUsers />}
                  >
                    Group Mentorship
                  </Button>
                </div>
              </AccordionItem>
            </Accordion>
          </ScrollShadow>
          <Button
            className="w-full mt-4"
            color="primary"
            startContent={<IconBolt />}
          >
            Upgrade to Premium
          </Button>
        </CardBody>
      </Card>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col md:flex-row">
        {/* Learning Content Pane */}
        <div className="flex-1 p-4 overflow-auto">
          <Tabs aria-label="Learning Content">
            <Tab key="overview" title="Overview">
              <Card>
                <CardBody>
                  <h2 className="text-2xl font-bold mb-4">Overview of GenAI</h2>
                  <p>
                    Generative AI (GenAI) refers to artificial intelligence
                    systems that can generate new content, including text,
                    images, audio, and more. These systems learn patterns from
                    existing data to create novel outputs that are similar in
                    style or content to the training data.
                  </p>
                </CardBody>
              </Card>
            </Tab>
            <Tab key="architecture" title="Architecture">
              <Card>
                <CardBody>
                  <h2 className="text-2xl font-bold mb-4">
                    GenAI Architecture
                  </h2>
                  <p>
                    GenAI systems typically use deep learning models,
                    particularly transformer architectures. These models process
                    input data through multiple layers of neural networks to
                    generate output.
                  </p>
                </CardBody>
              </Card>
            </Tab>
            <Tab key="benefits" title="Benefits">
              <Card>
                <CardBody>
                  <h2 className="text-2xl font-bold mb-4">Business Benefits</h2>
                  <p>
                    GenAI can enhance productivity, automate creative processes,
                    personalize customer experiences, and unlock new
                    possibilities in various industries.
                  </p>
                </CardBody>
              </Card>
            </Tab>
            <Tab key="examples" title="Examples">
              <Card>
                <CardBody>
                  <h2 className="text-2xl font-bold mb-4">GenAI Examples</h2>
                  <p>
                    Examples of GenAI include language models like GPT-3, image
                    generation models like DALL-E, and code generation tools
                    like GitHub Copilot.
                  </p>
                </CardBody>
              </Card>
            </Tab>
          </Tabs>
        </div>

        {/* Chat Interface Pane */}
        <Card className="w-full md:w-1/3 h-full rounded-none">
          <CardBody className="p-4 flex flex-col">
            <h2 className="text-2xl font-bold mb-4">AI Assistant</h2>
            <ScrollShadow className="flex-grow mb-4 p-4 rounded-md border">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`mb-4 ${
                    message.role === "user" ? "text-right" : "text-left"
                  }`}
                >
                  <span
                    className={`inline-block p-2 rounded-lg ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-default-100"
                    }`}
                  >
                    {message.content}
                  </span>
                </div>
              ))}
            </ScrollShadow>
            <div className="flex gap-2">
              <Input
                type="text"
                placeholder="Ask about AI skills..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
              />
              <Button
                isIconOnly
                onClick={handleSendMessage}
                aria-label="Send message"
              >
                <IconSend />
              </Button>
            </div>
            <div className="mt-2 text-sm text-default-400 flex items-center">
              <IconLock className="h-4 w-4 mr-1" />
              Advanced chat features require premium
            </div>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
