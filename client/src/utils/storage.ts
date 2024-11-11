type GeneratedContent = {
  sop?: string;
  rca?: string;
  timestamp: number;
};

export const storageKeys = {
  GENERATED_CONTENT: "incident-insights-content",
};

export const getStorageKey = (
  category: string,
  subcategory: string,
  symptom: string
) => {
  return `${category}|${subcategory}|${symptom}`;
};

export const saveGeneratedContent = (
  category: string,
  subcategory: string,
  symptom: string,
  type: "sop" | "rca",
  content: string
) => {
  try {
    const key = getStorageKey(category, subcategory, symptom);
    const storedData = localStorage.getItem(storageKeys.GENERATED_CONTENT);
    const contentMap: Record<string, GeneratedContent> = storedData
      ? JSON.parse(storedData)
      : {};

    contentMap[key] = {
      ...contentMap[key],
      [type]: content,
      timestamp: Date.now(),
    };

    localStorage.setItem(
      storageKeys.GENERATED_CONTENT,
      JSON.stringify(contentMap)
    );
  } catch (error) {
    console.error("Error saving to localStorage:", error);
  }
};

export const getGeneratedContent = (
  category: string,
  subcategory: string,
  symptom: string
): GeneratedContent | null => {
  try {
    const key = getStorageKey(category, subcategory, symptom);
    const storedData = localStorage.getItem(storageKeys.GENERATED_CONTENT);
    if (!storedData) return null;

    const contentMap: Record<string, GeneratedContent> = JSON.parse(storedData);
    return contentMap[key] || null;
  } catch (error) {
    console.error("Error reading from localStorage:", error);
    return null;
  }
};
