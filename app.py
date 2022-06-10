import streamlit as st
import pandas as pd 
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

##########################################################
# .streamlit/secrets.toml
def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True




#########################################################


st.set_page_config(page_title="Swarnandhra", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)


if check_password():
        hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        #logo = Image.open(r'swrnlogo.png')
        #import cv2
        #import plotly.express as px
        import io
        hide_menu_style = """
                <style>
                #MainMenu {visibility: hidden;}
                </style>
                """
        st.markdown(hide_menu_style, unsafe_allow_html=True)
        bootstrap="<link href='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css' rel='stylesheet' integrity='sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3' crossorigin='anonymous'>"
        st.markdown(bootstrap,unsafe_allow_html=True)
        with st.sidebar:
            choose = option_menu("Rainfall Predition Project 1", ["About", "KNN Algorithem", "Feature Scaling", "Random Forest Regression","Decision Tree Regression","Contact","Logout"],
                                icons=['house', 'bell', 'bell-fill','bell','bell', 'person lines fill','bell'],
                                menu_icon="award", default_index=0,
                                styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
            )





        if choose == "About":
          st.header("About")
          col1, col2 = st.columns( [0.2, 0.8])
          with col1:
              #st.image(logo)
              st.header("Welcome to My Project Rainfall Prediction")
          with col2:
            txt = '''
                <div class='card' style='width: 100%;'>
                
                <div class='card-body'>
                <h5 class='card-title'>Abstract:We introduce an architecture based on Deep Learning for the prediction of the accumlated daily precipitation for the next day.More specially,it includes an autoencoder for reducing and capturing non-linear relationship between attributes,and a multilayer perceptron for the prediction task.This arcitecture is compared with other previous proposals and it demonstrates an improvement on the ability to predict the accumulated daily precipitation for the next day</h5>
                
                </div>
                </div>
                '''
            st.markdown(txt,unsafe_allow_html=True)
            

        if choose == "KNN Algorithem":
                st.header("KKN algorithem:")
                uploaded_file = st.file_uploader("Choose a file")
                if uploaded_file is not None:
                    dataset = pd.read_csv(uploaded_file)
                    st.dataframe(dataset)
                    st.subheader("Data Set Loaded Successfully")
                    st.success("good")
                    #result of independentvariable
                    X = dataset.iloc[:,0:5].values # considering all station name
                    #st.write(X)
                    #result of dependent variable
                    y = dataset.iloc[:,36].values
                    #st.write(y)
                    #taking careof missing data
                    
                    labelencoder_X = LabelEncoder()
                    X[:,0]= labelencoder_X.fit_transform(X[:,0])
                    print("\n",X,"\n")

                    #y=y.astype(int)

                    X = np.array(X)
                    y = np.array(y)

                    #from sklearn.cross_validation import train_test_split
                    

                    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20,random_state = 33)


                    from sklearn.neighbors import KNeighborsRegressor
                    neigh = KNeighborsRegressor(n_neighbors=5)
                    neigh.fit(X_train,y_train)

                    predicted1 = neigh.predict(X_test)
                    print("Prediction Result: ",predicted1)

                    print('R-squared test score: {:.3f}'.format(neigh.score(X_test,y_test))) # R-Squared test score

                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize = (10, 5))
                    plt.plot(y_test,label='Actual')
                    plt.plot(predicted1,label='Predicted')
                    plt.legend()
                    
                    st.pyplot(fig)
                

        if choose == "Feature Scaling":
                st.header("Feature Scaling:")
                uploaded_file = st.file_uploader("Choose a file")
                if uploaded_file is not None:
                    dataset = pd.read_csv(uploaded_file)
                    st.dataframe(dataset)
                    st.subheader("Data Set Loaded Successfully")
                    st.success("good")

                    #result of independentvariable
                    X = dataset.iloc[:,0:5].values # considering all station name

                    #result of dependent variable
                    y = dataset.iloc[:,36].values

                    #taking careof missing data
                    from sklearn.preprocessing import LabelEncoder
                    labelencoder_X = LabelEncoder()
                    X[:,0]= labelencoder_X.fit_transform(X[:,0])
                    print("\n",X,"\n")
                    # Feature Scaling- MinMaxScaler
                    from sklearn import preprocessing
                    scaler = preprocessing.MinMaxScaler()
                    #X = scaler.fit_transform(X)
                    #y = scaler.fit_transform(y)

                    X = np.array(X)
                    y = np.array(y)

                    import sklearn
                    print (sklearn.__version__)

                    #!pip install -U scikit-learn
                    import sklearn
                    print (sklearn.__version__)
                    from sklearn.model_selection  import train_test_split

                    for i in range(1,100):
                        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20,random_state = i)
                        from sklearn.neighbors import KNeighborsRegressor
                        neigh = KNeighborsRegressor(n_neighbors=8)
                        neigh.fit(X_train,y_train)

                        predicted1 = neigh.predict(X_test)
                        #print("Prediction Result: ",predicted1)
                        print("i = ",i)
                        print('R-squared test score: {:.3f}'.format(neigh.score(X_test,y_test))) # R-Squared test score

                    import matplotlib.pyplot as plt
                    fig1 = plt.figure(figsize = (10, 5))
                    plt.plot(y_test,label='Actual')
                    plt.plot(predicted1,label='Predicted')
                    plt.legend()
                    plt.xlabel('Test Size')
                    plt.ylabel('Rainfall (mm)')
                    plt.title('Rainfall Prediction with KNN Regression')
                    plt.show()
                    st.pyplot(fig1)
        
        if choose == "Random Forest Regression":
                st.header("Random Forest Regression:")
                uploaded_file = st.file_uploader("Choose a file")
                if uploaded_file is not None:
                    dataset = pd.read_csv(uploaded_file)
                    st.dataframe(dataset)
                    st.subheader("Data Set Loaded Successfully")
                    st.success("good")
                    #result of independentvariable
                    X = dataset.iloc[:,0:5].values # considering all station name

                    #result of dependent variable
                    y = dataset.iloc[:,36].values

                    #taking careof missing data
                    from sklearn.preprocessing import LabelEncoder
                    labelencoder_X = LabelEncoder()
                    X[:,0]= labelencoder_X.fit_transform(X[:,0])
                    st.write("\n",X,"\n")

                    # Feature Scaling- MinMaxScaler
                    from sklearn import preprocessing
                    scaler = preprocessing.MinMaxScaler()
                    #X = scaler.fit_transform(X)
                    #y = scaler.fit_transform(y)

                    X = np.array(X)
                    y = np.array(y)

                    from sklearn.model_selection  import train_test_split
                    #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20,random_state = 33)


                    from sklearn.ensemble import RandomForestRegressor

                     #regressor = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=None, max_features=1, min_samples_leaf=1, min_samples_split=2, bootstrap=False)

                    for i in range(1,100):
                        #regressor = RandomForestRegressor(max_depth=None, random_state=i)
                        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20,random_state = i)
                        regressor = RandomForestRegressor(n_estimators=200, max_depth=None, max_features=1, min_samples_leaf=1, min_samples_split=2, bootstrap=False)
                        regressor.fit(X_train, y_train)

                        predicted1 = regressor.predict(X_test)
                        #print("Prediction Result: ",predicted1)
                        st.write("i = ",i)

                        st.write('R-squared test score: {:.3f}'.format(regressor.score(X_test,y_test))) # R-Squared test score

                    import matplotlib.pyplot as plt
                    fig2 = plt.figure(figsize = (10, 5))
                    plt.plot(y_test,label='Actual')
                    plt.plot(predicted1,label='Predicted')
                    plt.legend()
                    plt.xlabel('Test Size')
                    plt.ylabel('Rainfall (mm)')
                    plt.title('Rainfall Prediction with Random Forest Regression')
                    plt.show()
                    st.pyplot(fig2)
                
        if choose == "Decision Tree Regression":
                st.header("Decision Tree Regression:")
                uploaded_file = st.file_uploader("Choose a file")
                if uploaded_file is not None:
                    dataset = pd.read_csv(uploaded_file)
                    st.dataframe(dataset)
                    st.subheader("Data Set Loaded Successfully")
                    st.success("good")
                    #result of independentvariable
                    X = dataset.iloc[:,0:5].values # considering all station name

                    #result of dependent variable
                    y = dataset.iloc[:,36].values

                    #taking careof missing data
                    from sklearn.preprocessing import LabelEncoder
                    labelencoder_X = LabelEncoder()
                    X[:,0]= labelencoder_X.fit_transform(X[:,0])
                    st.write("\n",X,"\n")

                    # Feature Scaling- MinMaxScaler
                    from sklearn import preprocessing
                    scaler = preprocessing.MinMaxScaler()
                    #X = scaler.fit_transform(X)
                    #y = scaler.fit_transform(y)

                    X = np.array(X)
                    y = np.array(y)

                    from sklearn.model_selection  import train_test_split

                    from sklearn.tree import DecisionTreeRegressor
                    for i in range(1,100):
                        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20,random_state = i)
                        regressor = DecisionTreeRegressor(max_depth=6)
                        regressor.fit(X_train,y_train)

                        predicted1 = regressor.predict(X_test)
                        #print("Prediction Result: ",predicted1)
                        st.write('i =',i)
                        st.write('R-squared test score: {:.3f}'.format(regressor.score(X_test,y_test))) # R-Squared test score

                    import matplotlib.pyplot as plt
                    fig3 = plt.figure(figsize = (10, 5))
                    plt.plot(y_test,label='Actual')
                    plt.plot(predicted1,label='Predicted')
                    plt.legend()
                    plt.xlabel('Test Size')
                    plt.ylabel('Rainfall (mm)')
                    plt.title('Rainfall Prediction with Decision Tree Regression')
                    plt.show()
                    st.pyplot(fig3)

        if choose == "Contact":
          st.header("Contact Us:")
          st.success("Project Title:Raindall Prediction, Name:D.Dharani Satya, MCA Department")
          st.info("Email : dharanisatya123@gmail.com")
          st.warning("Phone: 9505930990")
        if choose == "Logout":
              del st.session_state['password_correct']  
              st.write("You are Logged Out..........")
              st.markdown("<a href='https://share.streamlit.io/ramabhadrarao/rainfallprediction/main/app.py' target='_self'>Click Here to Login Again</a>", unsafe_allow_html=True)  
