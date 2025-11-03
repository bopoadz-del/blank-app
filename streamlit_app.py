"""
Streamlit Application
A production-ready Streamlit app template with best practices.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os


# Page configuration
st.set_page_config(
    page_title="My Streamlit App",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main application function."""

    # Header
    st.title("🎈 My Streamlit App")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        app_mode = st.selectbox(
            "Choose a section:",
            ["Home", "Data Explorer", "Visualizations", "About"]
        )

        st.markdown("---")
        st.info(f"**Environment:** {os.getenv('ENVIRONMENT', 'development')}")
        st.info(f"**Version:** {os.getenv('APP_VERSION', '1.0.0')}")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Main content based on selection
    if app_mode == "Home":
        show_home()
    elif app_mode == "Data Explorer":
        show_data_explorer()
    elif app_mode == "Visualizations":
        show_visualizations()
    elif app_mode == "About":
        show_about()


def show_home():
    """Display the home page."""
    st.header("👋 Welcome!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Users",
            value="1,234",
            delta="12%"
        )

    with col2:
        st.metric(
            label="Active Sessions",
            value="567",
            delta="-3%"
        )

    with col3:
        st.metric(
            label="Response Time",
            value="23ms",
            delta="-5ms"
        )

    st.markdown("---")

    st.subheader("🚀 Getting Started")
    st.markdown("""
    This is a production-ready Streamlit application template with:

    - 📊 **Data Explorer**: Upload and analyze your data
    - 📈 **Visualizations**: Interactive charts and graphs
    - 🐳 **Docker Support**: Easy containerization
    - ✅ **Testing**: Built-in pytest integration
    - 🔄 **CI/CD**: GitHub Actions workflow
    - 📝 **Documentation**: Comprehensive README

    Navigate using the sidebar to explore different sections!
    """)

    # Quick start example
    with st.expander("🔥 Quick Example"):
        st.code("""
import streamlit as st
import pandas as pd

# Create a simple dataframe
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Tokyo']
})

# Display it
st.dataframe(df)
        """, language="python")


def show_data_explorer():
    """Display the data explorer page."""
    st.header("📊 Data Explorer")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=['csv'],
        help="Upload a CSV file to explore your data"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")

            # Display basic info
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

            with col2:
                st.subheader("ℹ️ Data Info")
                st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

                st.write("**Column Types:**")
                st.write(df.dtypes)

            # Statistics
            st.subheader("📈 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        # Generate sample data
        st.info("👆 Upload a CSV file or view the sample data below")

        sample_df = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'Value': np.random.randn(100).cumsum(),
            'Category': np.random.choice(['A', 'B', 'C'], 100)
        })

        st.dataframe(sample_df.head(20), use_container_width=True)


def show_visualizations():
    """Display the visualizations page."""
    st.header("📈 Visualizations")

    # Generate sample data
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['A', 'B', 'C']
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Line Chart")
        st.line_chart(chart_data)

    with col2:
        st.subheader("Area Chart")
        st.area_chart(chart_data)

    # Map example
    st.subheader("Map Visualization")
    map_data = pd.DataFrame(
        np.random.randn(100, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon']
    )
    st.map(map_data)


def show_about():
    """Display the about page."""
    st.header("ℹ️ About This App")

    st.markdown("""
    ### 🎈 Streamlit Production Template

    This is a comprehensive Streamlit application template designed for production use.

    #### 🛠️ Tech Stack
    - **Framework:** Streamlit
    - **Language:** Python 3.11
    - **Testing:** pytest
    - **Containerization:** Docker
    - **CI/CD:** GitHub Actions

    #### 📦 Features
    - ✅ Production-ready configuration
    - ✅ Docker and Docker Compose support
    - ✅ Automated testing with pytest
    - ✅ CI/CD pipeline
    - ✅ Environment variable management
    - ✅ Code formatting and linting
    - ✅ Comprehensive documentation

    #### 🔗 Links
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [GitHub Repository](https://github.com/)

    #### 📄 License
    MIT License - Feel free to use this template for your projects!
    """)

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")


if __name__ == "__main__":
    main()
