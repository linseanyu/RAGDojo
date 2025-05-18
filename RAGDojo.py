from langchain_community.document_loaders import YoutubeLoader

docs = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/shorts/CEoRmjLxE1c", add_video_info=True
).load()

docs[0].metadata