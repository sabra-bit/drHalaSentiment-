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
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y' , utc=True )
# data['Date'] = data['Date'].dt.strftime('%A %B %Y, %H:%M:%S')
data



dateList =['19-06-2024', '18-06-2024', '17-06-2024', '16-06-2024', '15-06-2024', '14-06-2024', '13-06-2024', '12-06-2024', '11-06-2024', '10-06-2024', '09-06-2024', '08-06-2024', '07-06-2024', '06-06-2024', '05-06-2024', '04-06-2024', '03-06-2024', '02-06-2024', '01-06-2024', '31-05-2024', '30-05-2024', '29-05-2024', '28-05-2024', '27-05-2024', '26-05-2024', '25-05-2024', '24-05-2024', '23-05-2024', '22-05-2024', '21-05-2024', '20-05-2024', '19-05-2024', '18-05-2024', '17-05-2024', '16-05-2024', '15-05-2024', '14-05-2024', '13-05-2024', '12-05-2024', '11-05-2024', '10-05-2024', '09-05-2024', '08-05-2024', '07-05-2024', '06-05-2024', '05-05-2024', '04-05-2024', '03-05-2024', '02-05-2024', '01-05-2024', '30-04-2024', '29-04-2024', '28-04-2024', '27-04-2024', '26-04-2024', '25-04-2024', '24-04-2024', '23-04-2024', '22-04-2024', '21-04-2024', '20-04-2024', '19-04-2024', '18-04-2024', '17-04-2024', '16-04-2024', '15-04-2024', '13-04-2024', '12-04-2024', '11-04-2024', '10-04-2024', '09-04-2024', '08-04-2024', '07-04-2024', '06-04-2024', '05-04-2024', '04-04-2024', '03-04-2024', '02-04-2024', '01-04-2024', '31-03-2024', '30-03-2024', '29-03-2024', '28-03-2024', '27-03-2024', '26-03-2024', '25-03-2024', '24-03-2024', '23-03-2024', '22-03-2024', '21-03-2024', '20-03-2024', '19-03-2024', '18-03-2024', '17-03-2024', '16-03-2024', '15-03-2024', '14-03-2024', '13-03-2024', '12-03-2024', '11-03-2024', '10-03-2024', '09-03-2024', '08-03-2024', '07-03-2024', '06-03-2024', '05-03-2024', '04-03-2024', '03-03-2024', '02-03-2024', '01-03-2024', '29-02-2024', '28-02-2024', '27-02-2024', '26-02-2024', '25-02-2024', '24-02-2024', '23-02-2024', '22-02-2024', '20-02-2024', '19-02-2024', '18-02-2024', '17-02-2024', '16-02-2024', '15-02-2024', '14-02-2024', '13-02-2024', '12-02-2024', '10-02-2024', '09-02-2024', '08-02-2024', '06-02-2024', '05-02-2024', '04-02-2024', '02-02-2024', '01-02-2024', '31-01-2024', '30-01-2024', '29-01-2024', '08-01-2024', '04-01-2024', '03-01-2024', '02-01-2024', '01-01-2024', '01-12-2023', '01-11-2023', '01-10-2023', '01-09-2023', '01-08-2023', '01-07-2023', '01-06-2023', '09-05-2023', '02-05-2023', '01-05-2023', '01-04-2023', '01-03-2023', '01-02-2023', '01-01-2023', '01-12-2022', '01-11-2022', '01-10-2022', '01-09-2022', '01-08-2022', '01-07-2022', '01-06-2022', '01-05-2022', '01-04-2022', '01-03-2022', '01-02-2022', '01-01-2022', '01-12-2021', '01-11-2021', '01-10-2021', '01-09-2021', '01-08-2021', '01-07-2021', '01-06-2021', '01-05-2021', '01-04-2021', '01-03-2021', '01-02-2021', '01-01-2021', '01-12-2020', '01-11-2020', '01-10-2020', '01-09-2020', '01-08-2020', '01-07-2020', '01-06-2020', '01-05-2020', '01-04-2020', '01-03-2020', '01-02-2020', '01-01-2020', '01-12-2019', '01-11-2019', '01-10-2019', '01-09-2019', '01-08-2019', '01-07-2019', '01-06-2019', '01-05-2019', '01-04-2019', '01-03-2019', '01-02-2019', '01-01-2019', '01-12-2018', '01-11-2018', '01-10-2018', '01-09-2018', '01-08-2018', '01-06-2018', '01-05-2018', '01-04-2018', '01-03-2018', '01-01-2018', '01-11-2017', '01-08-2017', '01-08-2016', '01-06-2016', '01-09-2015', '01-10-2010']
series = pd.Series(dateList )

# Convert dates to datetime objects
series = pd.to_datetime(series)

# Sort the Series
sorted_dates = series.sort_values().tolist()
list = []
print(sorted_dates)
for i in sorted_dates:
    list.append(str(i).split(' ')[0])
    

start_Date, end_Date = st.select_slider(
    "Select a range of Date",
    options=list,
    value=(list[0], list[-1]),
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


    replications = result_df['class'].value_counts()
    replications

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
    create_donut_chart(replications ,'class Distribution' )
    
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
    for class_name in options:
        try:
            # Filter data by class
            class_df = result_df[result_df['class'] == class_name]
            keyword_counts = class_df['keyWord'].value_counts().reset_index()
            keyword_counts.columns = ['keyWord', 'Count']

        

            sentment = class_df['Sentiment'].value_counts()
            
            # Prepare the text for the word cloud
            text = " ".join(class_df["translation"].astype(str))
            arabictext = " ".join(class_df["cleanText"].astype(str))
            # Tokenize and remove stop words
            words = nltk.word_tokenize(text)
            filtered_words = [word for word in words if word not in stop_words]

            # Create a new text string for the word cloud
            text = ' '.join(filtered_words)

            # Count word frequencies
            word_counts = Counter(text.split())

            # Generate word cloud for this class
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS, min_font_size=1).generate_from_frequencies(word_counts)
            st.subheader(f"Trending Words in {class_name} Class")
            st.code(arabictext)
            st.write("copy then follow link [arabic word cloud](https://worditout.com/word-cloud/create).")
            col1, col2, col3 = st.columns([3, 3, 1])
            with col1:
            # Display the word cloud
                # st.subheader(f"Trending Words in {class_name} Class")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                plt.figure(figsize=(8, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                
                plt.axis('off')
                plt.title(f"Trending Words in {class_name} Class")
                st.pyplot()
            with col2:
                
                create_donut_chart(sentment,f"sentment in {class_name} class")
            with col3:
                st.write(keyword_counts)
        except:
            
           st.write(f"no data in {class_name}")

        
            

    
    
    st.write("generation of visual network graphs  ")
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

