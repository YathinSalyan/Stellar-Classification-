import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Stellar Classification Project",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
    }
    .info-text {
        font-size: 1rem;
    }
    .highlight {
        background-color: #EFF6FF;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("<h1 class='main-header'>✨ Stellar Classification Project</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Explore and analyze star data using machine learning techniques</p>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.image("https://img.icons8.com/plasticine/100/000000/star.png", width=100)
st.sidebar.title("Navigation")
pages = ["Overview", "Data Exploration", "Correlation Analysis", "HR Diagram", "Machine Learning Models", "Predict Star Type"]
page = st.sidebar.radio("Go to", pages)

# Load data function
@st.cache_data
def load_data():
    try:
        return pd.read_csv("Stars.csv")
    except:
        # If file doesn't exist, create sample data for demonstration
        st.sidebar.warning("Actual data file not found. Using sample data for demonstration.")
        # Create sample data based on the project
        sample_data = {
            'Temperature': np.random.uniform(2000, 40000, 240),
            'L': np.random.uniform(0.001, 100000, 240),
            'R': np.random.uniform(0.01, 1000, 240),
            'A_M': np.random.uniform(-10, 20, 240),
            'Color': np.random.choice(['Red', 'Blue', 'White', 'Yellow', 'Yellowish', 'Orange'], 240),
            'Spectral_Class': np.random.choice(['M', 'B', 'A', 'F', 'G', 'K', 'O'], 240),
            'Type': np.random.randint(0, 6, 240)
        }
        df = pd.DataFrame(sample_data)
        return df

# Load the data
stars = load_data()

# Star type mapping
star_type_names = {
    0: "Brown Dwarf",
    1: "Red Dwarf",
    2: "White Dwarf",
    3: "Main Sequence",
    4: "Supergiant",
    5: "Hypergiant"
}

# Convert star types to names for display
stars['Type_Name'] = stars['Type'].map(star_type_names)

# Function to create the HR diagram
def create_hr_diagram(stars_data):
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['maroon', 'red', 'darkgray', 'deepskyblue', 'darkgoldenrod', 'navy']
    sizes = [30, 50, 75, 90, 100, 150]
    markers = ['.', '.', '.', '.', 'o', 'o']
    
    for star_type in range(6):
        subset = stars_data[stars_data['Type'] == star_type]
        ax.scatter(
            subset['Temperature'], 
            subset['A_M'],
            s=sizes[star_type],
            c=colors[star_type],
            marker=markers[star_type],
            label=star_type_names[star_type]
        )
    
    # Add the Sun position for reference
    ax.scatter(5778, 4.83, s=95, c='yellow', marker='o', label='Sun')
    
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Absolute Magnitude")
    ax.set_title("H-R Diagram of Stars")
    ax.legend()
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    return fig

# Function to preprocess data for ML
def preprocess_for_ml(data):
    # Create dummies for categorical variables
    spectral = pd.get_dummies(data['Spectral_Class'], prefix='Spectral')
    color = pd.get_dummies(data['Color'], prefix='Color')
    
    # Combine the data
    processed_data = pd.concat([
        data[['Temperature', 'L', 'R', 'A_M', 'Type']].reset_index(drop=True),
        spectral.reset_index(drop=True), 
        color.reset_index(drop=True)
    ], axis=1)
    
    return processed_data

# Function to train ML model
@st.cache_data
def train_model(data, model_type='rf'):
    processed_data = preprocess_for_ml(data)
    X = processed_data.drop('Type', axis=1)
    y = processed_data['Type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)
        
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    return model, X_train, X_test, y_train, y_test, y_pred, scaler

# Overview page
if page == "Overview":
    st.markdown("<h2 class='sub-header'>Project Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='highlight'>
        <p>This stellar classification project aims to analyze and categorize stars based on their physical properties 
        such as temperature, luminosity, radius, and absolute magnitude. The project utilizes various machine learning 
        techniques to predict star types from these measurements.</p>
        
        <p>Stars in this dataset are classified into six categories:</p>
        <ul>
            <li><strong>Brown Dwarf</strong>: Substellar objects that are too low in mass to sustain hydrogen fusion</li>
            <li><strong>Red Dwarf</strong>: Small, cool, main sequence stars</li>
            <li><strong>White Dwarf</strong>: Stellar remnants representing the final evolutionary state of stars</li>
            <li><strong>Main Sequence</strong>: "Normal" stars like our Sun that fuse hydrogen into helium</li>
            <li><strong>Supergiant</strong>: Extremely luminous stars with high mass</li>
            <li><strong>Hypergiant</strong>: The most massive and luminous stars known</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Display a pie chart of star types
        fig = px.pie(
            stars, 
            names='Type_Name', 
            title="Distribution of Star Types",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<h3 class='sub-header'>Dataset Information</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        - **Number of stars**: {len(stars)}
        - **Features**: Temperature, Luminosity, Radius, Absolute Magnitude, Color, Spectral Class
        - **Target**: Star Type
        """)
    
    with col2:
        # Count the number of each star type
        star_counts = stars['Type'].value_counts().reset_index()
        star_counts.columns = ['Type', 'Count']
        star_counts['Type_Name'] = star_counts['Type'].map(star_type_names)
        
        for index, row in star_counts.iterrows():
            st.markdown(f"**{row['Type_Name']}**: {row['Count']} stars")

    # Sample data display
    st.markdown("<h3 class='sub-header'>Sample Data</h3>", unsafe_allow_html=True)
    st.dataframe(stars.head())

# Data Exploration page
elif page == "Data Exploration":
    st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Feature Distributions", "Star Properties by Type", "3D Visualizations"])
    
    with tab1:
        st.markdown("### Distribution of Numerical Features")
        feature = st.selectbox("Select a feature to visualize:", 
                              ["Temperature", "L", "R", "A_M"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(
                stars, 
                x=feature, 
                color="Type_Name",
                title=f"Distribution of {feature}",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(
                stars, 
                y=feature, 
                color="Type_Name",
                title=f"Box Plot of {feature} by Star Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Display summary statistics
        st.markdown("### Summary Statistics")
        st.dataframe(stars[feature].describe())
    
    with tab2:
        st.markdown("### Star Properties by Type")
        
        # Select star type to analyze
        selected_type = st.selectbox(
            "Select star type:", 
            options=list(star_type_names.values())
        )
        
        # Filter data for selected type
        type_id = [k for k, v in star_type_names.items() if v == selected_type][0]
        type_data = stars[stars['Type'] == type_id]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {selected_type} Stars")
            st.markdown(f"Number of stars: **{len(type_data)}**")
            
            # Summary statistics
            st.dataframe(type_data[['Temperature', 'L', 'R', 'A_M']].describe())
        
        with col2:
            # Spectral class distribution
            spectral_counts = type_data['Spectral_Class'].value_counts().reset_index()
            spectral_counts.columns = ['Spectral_Class', 'Count']
            
            fig = px.bar(
                spectral_counts,
                x='Spectral_Class',
                y='Count',
                title=f"Spectral Classes for {selected_type} Stars",
                color='Spectral_Class'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Color distribution
        color_counts = type_data['Color'].value_counts().reset_index()
        color_counts.columns = ['Color', 'Count']
        
        fig = px.pie(
            color_counts,
            names='Color',
            values='Count',
            title=f"Color Distribution for {selected_type} Stars"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 3D Visualizations")
        
        # Options for axes
        axis_options = ["Temperature", "L", "R", "A_M", "Type"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-axis:", axis_options, index=0)
        with col2:
            y_axis = st.selectbox("Y-axis:", axis_options, index=1)
        with col3:
            z_axis = st.selectbox("Z-axis:", axis_options, index=2)
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            stars,
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color='Type_Name',
            title=f"3D Plot of {x_axis} vs {y_axis} vs {z_axis}",
            opacity=0.8
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Correlation Analysis page
elif page == "Correlation Analysis":
    st.markdown("<h2 class='sub-header'>Correlation Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Correlation Heatmap")
        
        # Correlation heatmap
        numeric_data = stars[["Temperature", "L", "R", "A_M"]]
        corr = numeric_data.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        ax.set_title("Correlation Between Features")
        st.pyplot(fig)
        
        st.markdown("""
        <div class='highlight'>
        <p><strong>Key Observations:</strong></p>
        <ul>
            <li>Absolute magnitude (A_M) is anti-correlated with other quantitative features</li>
            <li>Luminosity (L) is strongly correlated with radius (R)</li>
            <li>Temperature shows moderate correlation with luminosity</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Pairplot")
        
        # Create pairplot
        with st.spinner("Generating pairplot..."):
            fig = sns.pairplot(
                stars, 
                vars=["Temperature", "L", "R", "A_M"], 
                hue="Type_Name", 
                palette="Set2",
                height=1.8
            )
            st.pyplot(fig)

# HR Diagram page
elif page == "HR Diagram":
    st.markdown("<h2 class='sub-header'>Hertzsprung-Russell Diagram</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display HR diagram
        fig = create_hr_diagram(stars)
        st.pyplot(fig)
        
        # Add explanation
        st.markdown("""
        <div class='highlight'>
        <p>The Hertzsprung-Russell diagram is one of the most important tools in stellar astronomy. 
        It plots stars' surface temperature against their absolute magnitude (brightness).</p>
        
        <p>The diagram shows distinct groups of stars:</p>
        <ul>
            <li>Main sequence stars (including our Sun) form a diagonal band</li>
            <li>Giants and supergiants appear in the upper right region</li>
            <li>White dwarfs are found in the lower left</li>
        </ul>
        
        <p>This visualization helps astronomers understand stellar evolution and classify stars.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Star type selection
        st.markdown("### Filter by Star Type")
        selected_types = st.multiselect(
            "Select star types to display:",
            options=list(star_type_names.values()),
            default=list(star_type_names.values())
        )
        
        # Get type IDs from names
        type_ids = [k for k, v in star_type_names.items() if v in selected_types]
        filtered_stars = stars[stars['Type'].isin(type_ids)]
        
        # Display filtered HR diagram if types are selected
        if selected_types:
            fig = create_hr_diagram(filtered_stars)
            st.pyplot(fig)
        else:
            st.warning("Please select at least one star type.")
        
        # Sun position checkbox
        show_sun = st.checkbox("Show Sun position", value=True)
        
        if show_sun:
            st.markdown("""
            <p>The Sun is shown as a yellow dot on the diagram. It is a G-type main sequence star with:</p>
            <ul>
                <li>Temperature: 5778 K</li>
                <li>Absolute Magnitude: 4.83</li>
            </ul>
            """, unsafe_allow_html=True)

# Machine Learning Models page
elif page == "Machine Learning Models":
    st.markdown("<h2 class='sub-header'>Machine Learning Models</h2>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Model Performance", "Feature Importance"])
    
    with tab1:
        st.markdown("### Model Selection and Performance")
        
        model_type = st.radio(
            "Select model type:",
            options=["Random Forest", "Logistic Regression"],
            horizontal=True
        )
        
        # Train model based on selection
        selected_model = 'rf' if model_type == "Random Forest" else 'lr'
        model, X_train, X_test, y_train, y_test, y_pred, scaler = train_model(stars, selected_model)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display classification report
            st.markdown("#### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Overall accuracy
            accuracy = report_df.loc['accuracy', 'precision'] * 100
            st.markdown(f"**Overall Accuracy: {accuracy:.2f}%**")
        
        with col2:
            # Confusion matrix
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=[star_type_names[i] for i in range(6)],
                yticklabels=[star_type_names[i] for i in range(6)],
                ax=ax
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
    
    with tab2:
        if 'model' not in locals():
            model, X_train, X_test, y_train, y_test, y_pred, scaler = train_model(stars, 'rf')
        
        if selected_model == 'rf':
            st.markdown("### Feature Importance")
            
            # Get feature importance
            feature_importance = model.feature_importances_
            feature_names = X_train.columns
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(
                importance_df.head(15),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Most Important Features',
                color='Importance'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class='highlight'>
            <p><strong>Interpretation:</strong></p>
            <p>Feature importance shows which stellar properties are most useful for classification. 
            This can provide insight into the physical properties that best distinguish between different types of stars.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Feature importance is only available for the Random Forest model. Please select Random Forest in the Model Performance tab.")

# Predict Star Type page
elif page == "Predict Star Type":
    st.markdown("<h2 class='sub-header'>Predict Star Type</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight'>
    <p>Enter the stellar properties below to predict the star type using our Random Forest model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Train model
    model, X_train, X_test, y_train, y_test, y_pred, scaler = train_model(stars, 'rf')
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Temperature (K)", 2000, 40000, 5778)
        luminosity = st.slider("Luminosity (L☉, logarithmic)", -4.0, 6.0, 0.0)
        # Convert from log scale to linear for the model
        luminosity_linear = 10 ** luminosity
    
    with col2:
        radius = st.slider("Radius (R☉, logarithmic)", -2.0, 3.0, 0.0)
        # Convert from log scale to linear for the model
        radius_linear = 10 ** radius
        absolute_magnitude = st.slider("Absolute Magnitude", -10.0, 20.0, 4.83)
    
    # Additional categorical features
    spectral_class = st.selectbox("Spectral Class", options=['O', 'B', 'A', 'F', 'G', 'K', 'M'])
    star_color = st.selectbox("Color", options=['Blue', 'White', 'Yellow', 'Orange', 'Red'])
    
    # Predict button
    if st.button("Predict Star Type"):
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'Temperature': [temperature],
            'L': [luminosity_linear],
            'R': [radius_linear],
            'A_M': [absolute_magnitude],
            'Spectral_Class': [spectral_class],
            'Color': [star_color]
        })
        
        # Preprocess the input data
        # Create dummy variables for spectral class and color
        spectral_dummies = pd.get_dummies(input_data['Spectral_Class'], prefix='Spectral')
        color_dummies = pd.get_dummies(input_data['Color'], prefix='Color')
        
        # Drop original categorical columns
        input_data = input_data.drop(['Spectral_Class', 'Color'], axis=1)
        
        # Combine all features
        processed_input = pd.concat([input_data, spectral_dummies, color_dummies], axis=1)
        
        # Add missing columns that might be in the training data but not in the input
        for col in X_train.columns:
            if col not in processed_input.columns:
                processed_input[col] = 0
        
        # Ensure columns are in the same order as during training
        processed_input = processed_input[X_train.columns]
        
        # Scale the input data
        scaled_input = scaler.transform(processed_input)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]
        
        # Display prediction
        st.markdown(f"""
        <div style="background-color: #EFF6FF; padding: 20px; border-radius: 10px; text-align: center;">
            <h3>Prediction Result</h3>
            <h2 style="color: #1E3A8A;">{star_type_names[prediction]}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display confidence levels
        st.markdown("### Prediction Confidence")
        
        # Create DataFrame for confidence levels
        confidence_df = pd.DataFrame({
            'Star Type': [star_type_names[i] for i in range(6)],
            'Confidence': prediction_proba
        })
        
        # Plot confidence levels
        fig = px.bar(
            confidence_df,
            x='Star Type',
            y='Confidence',
            color='Confidence',
            color_continuous_scale='Blues',
            title='Confidence Levels for Each Star Type'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # HR Diagram with the predicted star position
        st.markdown("### Position on HR Diagram")
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot all stars with reduced opacity
        for star_type in range(6):
            subset = stars[stars['Type'] == star_type]
            ax.scatter(
                subset['Temperature'], 
                subset['A_M'],
                s=30,
                c=plt.cm.Set2(star_type),
                alpha=0.3,
                label=star_type_names[star_type] if star_type == prediction else None
            )
        
        # Plot the predicted star with a larger marker
        ax.scatter(
            temperature, 
            absolute_magnitude,
            s=200,
            c=plt.cm.Set2(prediction),
            marker='*',
            edgecolors='black',
            label='Your Star'
        )
        
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Absolute Magnitude")
        ax.set_title("Your Star's Position on the HR Diagram")
        ax.legend(loc='best')
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        st.pyplot(fig)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px;">
    <p>Stellar Classification Project | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)