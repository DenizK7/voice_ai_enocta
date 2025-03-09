# app/models/transcript.py
class Transcript:
    def __init__(self, video_url: str, start: float, end: float, text: str):
        self.video_url = video_url
        self.start = start
        self.end = end
        self.text = text

    def to_dict(self):
        return {
            "video_url": self.video_url,
            "start": self.start,
            "end": self.end,
            "text": self.text
        }
