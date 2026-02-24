from urllib.parse import urlparse, parse_qs
import os
import html


def extract_video_id(url):
    parsed_url = urlparse(url)

    if "youtube.com" in parsed_url.netloc:
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]

    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path[1:]

    return None


def get_transcript(video_id):
    """
    Fetches the captions/transcript for a YouTube video using the
    official YouTube Data API v3 (captions resource).

    Returns (transcript_text, error_message).
    transcript_text is None only if the fetch fails.

    Required env var:
        YOUTUBE_API_KEY  — your Google Cloud YouTube Data API v3 key
    """
    api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not api_key:
        return None, (
            "YOUTUBE_API_KEY is not set. "
            "Add it to your .env file or environment variables."
        )

    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

        youtube = build("youtube", "v3", developerKey=api_key)

        # ── 1. Try the official captions list + download ──────────────────────
        try:
            captions_response = youtube.captions().list(
                part="snippet",
                videoId=video_id
            ).execute()

            caption_items = captions_response.get("items", [])

            # Prefer English captions; fall back to the first available track
            caption_id = None
            for item in caption_items:
                lang = item["snippet"].get("language", "")
                if lang.startswith("en"):
                    caption_id = item["id"]
                    break

            if caption_id is None and caption_items:
                caption_id = caption_items[0]["id"]

            if caption_id:
                raw_bytes = youtube.captions().download(
                    id=caption_id,
                    tfmt="srt"          # SubRip format — plain text segments
                ).execute()

                transcript_text = _parse_srt(raw_bytes.decode("utf-8", errors="replace"))

                if transcript_text and transcript_text.strip():
                    return transcript_text, None

        except HttpError as caption_err:
            # 403 / 401 means OAuth is required for caption download — fall through
            # to the transcript3 fallback below
            if caption_err.resp.status not in (401, 403):
                raise

        # ── 2. Fallback: youtube-transcript-api (pip package) ─────────────────
        #    This works for videos that have auto-generated or manual captions
        #    with public visibility, without needing OAuth.
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["en", "en-US", "en-GB"]
            )
            transcript_text = " ".join(
                html.unescape(seg["text"]) for seg in transcript_list
            )

            if transcript_text and transcript_text.strip():
                return transcript_text, None

        except Exception:
            pass   # Will try any language next

        # ── 3. Last resort: any available language via transcript API ──────────
        try:
            from youtube_transcript_api import YouTubeTranscriptApi, TranscriptList

            transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list_obj.find_transcript(
                [t.language_code for t in transcript_list_obj]
            )
            segments = transcript.fetch()
            transcript_text = " ".join(
                html.unescape(seg["text"]) for seg in segments
            )

            if transcript_text and transcript_text.strip():
                return transcript_text, None

        except Exception:
            pass

        # ── 4. Pull video metadata as fallback context ─────────────────────────
        #    If no caption track is accessible, return the video description,
        #    which at least gives the RAG pipeline something to work with.
        video_response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        items = video_response.get("items", [])
        if not items:
            return None, f"YouTube API returned no data for video ID '{video_id}'."

        snippet = items[0]["snippet"]
        description = snippet.get("description", "").strip()
        title = snippet.get("title", "")

        if not description:
            return None, (
                "No transcript or description is available for this video. "
                "It may be private, restricted, or have no captions enabled."
            )

        metadata_text = f"Title: {title}\n\nDescription:\n{description}"
        return metadata_text, (
            "Note: A full transcript could not be retrieved. "
            "Answers are based on the video's title and description only."
        )

    except Exception as e:
        return None, f"YouTube Data API fetch failed: {e}"


# ── Internal helpers ────────────────────────────────────────────────────────

def _parse_srt(srt_text):
    """
    Strips SRT timing/index lines and returns a clean block of caption text.
    """
    import re
    lines = srt_text.splitlines()
    text_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.isdigit():               # sequence number
            continue
        if re.match(r"\d{2}:\d{2}:\d{2}", line):   # timestamp line
            continue
        text_lines.append(html.unescape(line))
    return " ".join(text_lines)
