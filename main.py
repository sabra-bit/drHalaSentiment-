import streamlit as st
import openpyxl
import pandas as pd
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
import plotly.graph_objects as go
st.title('arabic text analysis')
workbook = openpyxl.load_workbook(filename="allData1.xlsx", data_only=True)
sheet = workbook.active
data = []
for row in sheet.iter_rows():
  row_data = []
  for cell in row:
    row_data.append(cell.value)
  data.append(row_data)
data = pd.DataFrame(data)
data = data.dropna()
new_column_names = {0:'#',1:'scrapedData',2:'Date',3:'cleanText',4: 'class', 5:'keyWord',6:'Sentiment'}

data = data.rename(columns=new_column_names)


data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce', utc=True)

data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce', utc=True)
# data['Date'] = data['Date'].dt.strftime('%A %B %Y, %H:%M:%S')
data



data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

# Step 2: Drop missing/invalid dates
valid_dates = data['Date'].dropna()

# Step 3: Get unique dates and sort them
sorted_dates = valid_dates.drop_duplicates().sort_values()

# Step 4: Convert to string format (e.g., 'YYYY-MM-DD' or any you prefer)
date_string_list = sorted_dates.dt.strftime('%Y-%m-%d').tolist()




    

start_Date, end_Date = st.select_slider(
    "Select a range of Date",
    options=date_string_list,
    value=(date_string_list[0], date_string_list[-1]),
)


start_date = pd.to_datetime(start_Date).tz_localize('UTC')
end_date = pd.to_datetime(end_Date).tz_localize('UTC')

filtered_df = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
filtered_df

options = st.multiselect(
    "choose the class",
    ["Enviromental", "Economic", "Social", "unclassified"],
    
)
if options:
    result_df = filtered_df[filtered_df['class'].isin(options)]
    result_df
    
    sentiment_counts = result_df.groupby(['class', 'Sentiment']).size().unstack(fill_value=0)
    # sentiment_counts
    replications = result_df['class'].value_counts()
    # replications

    def create_donut_chart(replications, title):
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Custom autopct function to show count and percentage
        def autopct_format(pct, allvals):
            total = sum(allvals)
            count = int(round(pct * total / 100.0))
            return f"{pct:.1f}% ({count}) "

        fig, ax = plt.subplots(figsize=(4, 4))  # Smaller size
        wedges, texts, autotexts = ax.pie(
            replications,
            labels=replications.index,
            autopct=lambda pct: autopct_format(pct, replications),
            startangle=140,
            colors=plt.cm.Pastel1.colors,
            wedgeprops={'linewidth': 3, 'edgecolor': 'white'}
        )

        # Add donut hole
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)

        ax.axis('equal')
        plt.title(title, fontsize=10)

        st.pyplot(fig)

    
    replication_df = replications.reset_index()
    replication_df.columns = ['Class', 'Count']
    replication_df['Percentage'] = (replication_df['Count'] / replication_df['Count'].sum()) * 100

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=replication_df['Class'],
            y=replication_df['Count'],
            text=[f"{count} ({pct:.1f}%)" for count, pct in zip(replication_df['Count'], replication_df['Percentage'])],
            textposition='auto',
            marker_color='lightblue'
        )
    ])

    fig.update_layout(
        title="Class Distribution",
        xaxis_title="Class",
        yaxis_title="Count",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    

    
    cols = st.columns(len(replication_df))
    total = replications.sum()
    for i, (_, row) in enumerate(replication_df.iterrows()):
        class_name = row['Class']
        count = row['Count']
        percentage = (count / total) * 100
        cols[i].metric(label=class_name, value=f"{count}", delta=f"{percentage:.1f}%")
   

       
            
    def create_translation_dict(file_path, sheet_name):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df = df[['word', 'translate']]
        return df.set_index('word')['translate'].to_dict()

    # Load the translation data into a dictionary
    translation_dict = create_translation_dict("translate.xlsx", "Sheet1")

    def translate_text(text):
        return translation_dict.get(text, text)  # Default to original text if not found

    # Apply the translation function to the 'keyword' column
    result_df['translation'] = result_df['keyWord'].dropna().apply(translate_text)




    result_df['translation'] = result_df['translation'].str.lower()
    # Prepare text data
    text = " ".join(result_df["translation"].astype(str))
    
    from collections import Counter
    import nltk
    nltk.download('stopwords') 
    from nltk.corpus import stopwords

    # ... your code
    nltk.download('punkt')
    # Tokenize text
    words = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    on = st.toggle("English word frequency")
    wordkey = 'keyWord'
    if on:
        wordkey = 'translation'
    for class_name in options:
        try:
            # Filter data by class
            
           
            class_df = result_df[result_df['class'] == class_name]
            keyword_counts = class_df[wordkey].value_counts().reset_index().head(10)
            keyword_counts.columns = [wordkey, 'Count']
            keyword_counts = keyword_counts.set_index(wordkey)
           
        
            sentment = class_df['Sentiment'].value_counts()
            # class_df
            # Prepare the text for the word cloud
            text = " ".join(class_df["translation"].astype(str))
            arabictext = " ".join(class_df["keyWord"].astype(str))
            # Tokenize and remove stop words
            words = nltk.word_tokenize(text)
            filtered_words = [word for word in words if word not in stop_words]

            # Create a new text string for the word cloud
            text = ' '.join(filtered_words)

            # Count word frequencies
            word_counts = Counter(text.split())

            # Generate word cloud for this class
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS, min_font_size=1).generate_from_frequencies(word_counts)
            st.subheader(f"Trending Words in :blue[{class_name}] Class")
            st.code(arabictext)
            st.write("copy then follow link [arabic word cloud](https://worditout.com/word-cloud/create).")
            col1, col2, col3 = st.columns([3,2, 2])
            with col1:
            # Display the word cloud
                st.subheader("word cloud", divider=True)
                # st.subheader(f"Trending Words in {class_name} Class")
                # st.set_option('deprecation.showPyplotGlobalUse', False)
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f"Trending Words in {class_name} Class")
                st.pyplot(fig)  # ✅ آمن ومعتمد

            with col2:
                
                sentiment_df = sentment.reset_index()
                sentiment_df.columns = ['Sentiment', 'Count']
                rows = [sentiment_df.iloc[i:i+2] for i in range(0, len(sentiment_df), 2)]
                st.subheader("sentment", divider=True)
                for row in rows:
                    cols = st.columns(2)
                    for i, (_, data) in enumerate(row.iterrows()):
                        sentiment = data['Sentiment']
                        count = data['Count']
                        total = sentment.sum()
                        percentage = (count / total) * 100
                        cols[i].metric(label=sentiment, value=f"{count}", delta=f"{percentage:.1f}%")


                # create_donut_chart(sentment,f"sentment in {class_name} class")
            with col3:
                st.subheader("word frequency", divider=True)
                keyword_counts['Percentage'] = (keyword_counts['Count'] / keyword_counts['Count'].sum()) * 100
                fig = go.Figure(data=[
                go.Bar(
                    x=keyword_counts.index,
                    y=keyword_counts['Count'],
                    text=[f"{count} ({pct:.1f}%)" for count, pct in zip(keyword_counts['Count'], keyword_counts['Percentage'])],
                    textposition='auto',
                    marker_color='indianred'
                        )
                    ])
                fig.update_layout(
                title=f"Top Keywords in {class_name} Class",
                xaxis_title="Keyword",
                yaxis_title="Count",
                height=400
                 )

                st.plotly_chart(fig, use_container_width=True)
                # st.bar_chart(keyword_counts['Count'])
        except Exception as e:
            
            st.write(f"{e}")

        
            

    
    st.subheader("generation of visual network graphs  ", divider=True)
    
    # Create networkx graph object from pandas dataframe
    G =  nx.MultiGraph()
    
    G.add_edge('Data',"Enviromental")
    G.add_edge('Data',"Economic")
    G.add_edge('Data',"Social")
    # G.add_node('alex', size=10, title='chile',color='green',)
    datashow  =result_df
    result_df = result_df.groupby(['class', 'keyWord']).size().reset_index(name='count')
    for index, row in result_df.iterrows():
        
        G.add_node(str(row['keyWord']), size=20, title=str(row['count']),)
        G.add_edge(str(row['class']),str(row['keyWord']))
                
    # G.add_node('Data', size=20, title='head',color='red',)

    # pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    # nx.draw(G,pos=pos,with_labels=True,node_size=1000)

        # Initiate PyVis network object
    drug_net = Network(height='835px', width='835px',directed=True, bgcolor='#222222', font_color='white' )

        # Take Networkx graph and translate it to a PyVis graph format
    drug_net.from_nx(G)

        # Generate network with specific layout settings
    drug_net.repulsion(node_distance=520, central_gravity=0.7,spring_length=110, spring_strength=0.60,damping=0.13,)
    
    drug_net.show_buttons(filter_=['physics'])


        # Save and read graph as HTML file (on Streamlit Sharing)
    try:
            path = 'img'
            drug_net.save_graph(f'{path}\\pyvis_graph.html')
            HtmlFile = open(f'{path}\\pyvis_graph.html', 'r', encoding='utf-8')

        # Save and read graph as HTML file (locally)
    except:
            path = 'html_files'
            drug_net.save_graph(f'{path}\\pyvis_graph.html')
            HtmlFile = open(f'{path}\\pyvis_graph.html', 'r', encoding='utf-8')
    
    
    components.html(HtmlFile.read(), height=835 ,width=835,scrolling=True)

    datashow
    # result_df

