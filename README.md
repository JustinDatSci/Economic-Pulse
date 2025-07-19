# 🇺🇸 Economic Pulse Dashboard

A real-time, interactive dashboard for visualizing key U.S. economic indicators using data from the FRED API and generating AI-powered analysis with the OpenAI API.

---

### **Live Demo**

**[View Live App](https://economic-pulse.streamlit.app/)**

---

### **Screenshot**

![Economic Pulse Screenshot](https://github.com/user-attachments/assets/6706740d-22d3-4d41-bd2e-11380f6e9f4e)

---

### **Features**

**Dynamic Dashboard**: Select which economic indicators to display from the sidebar for a customized view.

**Interactive Charts**: Visualize data with responsive Plotly charts that update based on your selections.

**AI-Powered Analysis**: Generate a concise summary of the current economic climate with the click of a button using the OpenAI API.

**Performance Optimized**: Caches API calls for both data and AI summaries to ensure fast load times and efficient resource use.

**Custom Styling**: Features a unique "pulse" animation on the title using custom CSS.

---

### **Tech Stack**

**Language**: Python

**Framework**: Streamlit

**Key Libraries**: Pandas, Plotly, FredAPI, OpenAI

**Deployment**: Streamlit Community Cloud

---

### **⚙️ How to Run Locally**

To run this project on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/JustinDatSci/Economic-Pulse.git](https://github.com/JustinDatSci/Economic-Pulse.git)
    cd Economic-Pulse
    ```

2.  **Create a `.env` file** in the root directory and add your API keys:
    ```
    FRED_API_KEY="YOUR_FRED_API_KEY"
    OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
