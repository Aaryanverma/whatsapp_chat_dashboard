import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
import io
import time
from collections import Counter
from datetime import datetime
import plotly.express as px
from google_trans_new import google_translator
from multiprocessing.dummy import Pool as ThreadPool
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="WhatsApp Chat Dashboard",
    page_icon="🔍",
    layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title("WhatsApp Chat Dashboard")
st.markdown('<small>Made with ♥ in India. © <b>Aaryan Verma</b></small>',unsafe_allow_html=True)

translator = google_translator()

sid_obj = SentimentIntensityAnalyzer()
pool = ThreadPool(8)
stopwords = set(STOPWORDS)

with st.beta_expander("How to export your Conversation"):
    st.write("""To export a copy of the history of an individual chat or group:
  \n1. Open the conversation or group.
  \n2. For Android: Click on three vertical dots on top right corner and select More.  \nFor iOS: Tap on Contact/Group Name.
  \n3. Select Export chat.
  \n4. Choose Without Media.
  \n5. You will asked how to save your chat history attached as a .txt document.  \nSave it wherever you like. Then download the .txt file and upload it below.""")

chat_file = st.file_uploader("Upload chat file (Don't worry your data is safe. Analysis is done in your browser only and won't be uploaded anywhere.)", type=["txt"])

chat_content = []

if chat_file != None:
    raw_text = io.TextIOWrapper(chat_file,encoding='utf-8')
    chat_content = raw_text.readlines()

def translate_request(text):
    try:
        translate_text = translator.translate(text.strip().lower(), lang_tgt='en')
        time.sleep(0.5)
        translate_text = " ".join(word for word in translate_text if word not in stopwords)
        return translate_text
    except:
        st.error("Too many requests for Sentiment Analyzer. Pls, Try again after some time.")


def list_to_DF(_list,f=0):

    d_t_format=['%d/%m/%Y, %I:%M %p','%d/%m/%y, %I:%M %p','%m/%d/%y, %I:%M %p']
    date=re.compile('\d{1,2}/\d{1,2}/\d{2,4}')

    df=pd.DataFrame(columns=['date_time','author','message'])
    for chat in _list:
        if date.match(chat):
            datetym,conversation=re.split('-',chat,maxsplit=1)
            try:
                aut,msg=re.split(':',conversation,maxsplit=1)
            except ValueError:
                aut=np.nan
                msg=str.strip(conversation)
            d=str.strip(datetym)
            try:
                d_t=datetime.strptime(str.strip(datetym),d_t_format[f])
            except ValueError:
                return list_to_DF(_list,f+1)
            df=df.append({'date_time':d_t,'author':aut,'message':str.strip(msg)},ignore_index=True)
        else:
            df.iloc[-1].message=df.iloc[-1].message+' '+chat

    return df

def data_preperation(df):

    y = lambda x:x.year
    emg_extrct = lambda x:''.join(re.findall(emoji.get_emoji_regexp(),x))
    count_w = lambda x:len(x.split())
    count_emoji = lambda x:len(list(x))
    URLPATTERN = r'(https?://\S+)'

    df.dropna(inplace=True)
    df['date'] = df['date_time'].apply(pd.Timestamp.date)
    df['day'] = df['date_time'].apply(pd.Timestamp.day_name)
    df['month'] = df['date_time'].apply(pd.Timestamp.month_name)
    df['year'] = df['date_time'].apply(y)    #(pd.Timestamp.year)
    df['time'] = df['date_time'].apply(pd.Timestamp.time).apply(lambda x: datetime.strptime(str(x), "%H:%M:%S")).apply(lambda x: x.strftime("%I:%M %p"))
    df['emoji_used'] = df.message.apply(emg_extrct)
    df['word_count'] = df.message.apply(count_w)
    df['emoji_count'] = df.emoji_used.apply(count_emoji)
    df['Media'] = df.message.str.contains('<Media omitted>')
    df['urlcount'] = df.message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
    return df

if chat_content!=[]:
    df=list_to_DF(chat_content)
    df=data_preperation(df)

    st.subheader("Group Wise Stats")
    st.write("\n")
    st.write("Total Text Messages: ", df.shape[0])
    st.write("Total Media Messages: ", df[df['Media']].shape[0])
    st.write("Total Emojis: ", sum(df['emoji_used'].str.len()))
    st.write("Total Links/URLs: ", np.sum(df.urlcount))

    media_messages_df = df[df['message'] == '<Media omitted>']
    messages_df = df.drop(media_messages_df.index)

    author_value_counts = df['author'].value_counts().to_frame()
    fig0 = px.bar(author_value_counts, y='author', x=author_value_counts.index, labels={'index':'Participants','author':'messages count'}, title="Top Chatter")
    st.plotly_chart(fig0)

    sort_type = st.selectbox("Sort By:",["Date","Day","Time","Month"])
    if sort_type=="Date":
        keyword="date"
    elif sort_type=="Day":
        keyword="day"
    elif sort_type=="Time":
        keyword = "time"
    elif sort_type=="Month":
        keyword = "month"

    sort_df = messages_df.groupby(keyword).sum()
    sort_df['MessageCount'] = messages_df.groupby(keyword).size().values
    sort_df.reset_index(inplace=True)
    fig = px.line(sort_df, x=keyword, y="MessageCount", title=f"Number of Messages according to {keyword}",)
    fig.update_xaxes(nticks=20,showgrid=False)
    st.plotly_chart(fig)

    # emoji distribution
    senders = st.selectbox("Select participant:",messages_df.author.unique())
    dummy_df = messages_df[messages_df['author'] == senders]
    total_emojis_list = list([a for b in dummy_df.emoji_used for a in b])
    emoji_dict = dict(Counter(total_emojis_list))
    emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
    author_emoji_df = pd.DataFrame(emoji_dict, columns=['emoji', 'count'])
    fig5 = px.pie(author_emoji_df, values='count', names='emoji', title=f'Emoji Distribution for {senders}')
    fig5.update_traces(textposition='inside', textinfo='percent+label',showlegend=False)
    st.plotly_chart(fig5)

    
    comment_words = ''

    for val in dummy_df.message:
        # typecaste each val to string
        val = str(val)
        # split the value
        tokens = val.split()
        # Converts each token into lowercase
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='black',
                stopwords = stopwords,min_font_size=6).generate(comment_words)
    
    # plot the WordCloud image
    with st.beta_expander("Tap to View Wordcloud"):       
        fig, ax = plt.subplots(figsize = (10, 10),facecolor = 'k')
        ax.imshow(wordcloud,interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig)

    senti = []
    with st.spinner(f'Analyzing Sentiment for {senders}.. (This may take some time depending on number of messages)'):
        try:
            translation = pool.map(translate_request, dummy_df["message"].values)
        except Exception as e:
            raise e
        pool.close()
        pool.join()
        

        for i in translation:
            sentiment_dict = sid_obj.polarity_scores(i)
            if sentiment_dict['compound'] >= 0.05 :
                senti.append("Positive")
            elif sentiment_dict['compound'] <= - 0.05 :
                senti.append("Negative")
            else :
                senti.append("Neutral")
    
    all_sents = Counter(senti)
    fig6 = px.bar(y=all_sents.values(), x=all_sents.keys(), labels={'x':'Sentiment','y':'Messages'},title=f"Sentiments for {senders}")
    st.plotly_chart(fig6)
    result = max(all_sents,key=all_sents.get)
    st.info(f"{senders} mostly sends {result} messages")

st.markdown('  <br><br><center>Developed and Maintained by\
             <b><a href="https://www.linkedin.com/in/aaryanverma" target="_blank">Aaryan Verma</a></b></center>',unsafe_allow_html=True)
