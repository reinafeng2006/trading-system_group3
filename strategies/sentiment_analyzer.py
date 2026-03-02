"""
Media Sentiment Analyzer

Fetches news from Finnhub API and Google News RSS, analyzes with FinBERT.

Weighting:
- Finnhub Company News: 50%
- Finnhub Market News: 30%
- Google News RSS: 20%

Usage:
    python sentiment_analyzer.py AAPL
"""

import requests
from bs4 import BeautifulSoup
import feedparser
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import re
from urllib.parse import quote
import time

FINNHUB_API_KEY = "d67n5f1r01qobepik77gd67n5f1r01qobepik780"


class MediaSentimentAnalyzer:

    def __init__(
        self,
        finnhub_api_key: str = None,
        recency_decay_lambda: float = 0.1,
        max_company_news: int = 10,
        max_market_news: int = 5,
        max_google_news: int = 5,
        company_news_weight: float = 0.50,
        market_news_weight: float = 0.30,
        google_news_weight: float = 0.20
    ):
        self.finnhub_api_key = finnhub_api_key or FINNHUB_API_KEY

        if not self.finnhub_api_key or self.finnhub_api_key == "PASTE_YOUR_API_KEY_HERE":
            raise ValueError(
                "Finnhub API key not set!\n"
                "Get a free API key at: https://finnhub.io/register"
            )

        self.decay_lambda = recency_decay_lambda
        self.max_company_news = max_company_news
        self.max_market_news = max_market_news
        self.max_google_news = max_google_news
        self.company_news_weight = company_news_weight
        self.market_news_weight = market_news_weight
        self.google_news_weight = google_news_weight
        self.finnhub_base = "https://finnhub.io/api/v1"
        self._finbert = None
        self._finbert_tokenizer = None

    def _load_models(self):
        if self._finbert is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print("Loading FinBERT...")
            self._finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self._finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            print("FinBERT loaded!\n")

    def _naive(self, dt: datetime) -> datetime:
        """Strip timezone info so all comparisons are between naive datetimes."""
        return dt.replace(tzinfo=None)

    def fetch_finnhub_company_news(self, symbol: str, hours_back: int = 24, reference_time: datetime = None) -> List[Dict]:
        print(f"Fetching Finnhub company news for {symbol}...")
        if reference_time is None:
            reference_time = datetime.now()
        reference_time = self._naive(reference_time)

        to_date = reference_time
        from_date = to_date - timedelta(hours=hours_back)

        url = f"{self.finnhub_base}/company-news"
        params = {
            'symbol': symbol,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'token': self.finnhub_api_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                print(f"  Error: Status {response.status_code}")
                return []

            data = response.json()
            articles = []
            for item in data[:self.max_company_news]:
                pub_time = self._naive(datetime.fromtimestamp(item['datetime']))
                text = self._fetch_article_text(item['url'], item['headline'])
                if not text or len(text) < 100:
                    text = f"{item['headline']}. {item.get('summary', '')}"
                articles.append({
                    'title': item['headline'],
                    'url': item['url'],
                    'published': pub_time,
                    'text': text,
                    'summary': item.get('summary', ''),
                    'source': 'finnhub_company'
                })

            print(f"  Found {len(articles)} company news articles")
            return articles

        except Exception as e:
            print(f"  Error fetching Finnhub company news: {e}")
            return []

    def fetch_finnhub_market_news(self, hours_back: int = 24, reference_time: datetime = None) -> List[Dict]:
        print(f"Fetching Finnhub market news...")
        if reference_time is None:
            reference_time = datetime.now()
        reference_time = self._naive(reference_time)

        cutoff_time = reference_time - timedelta(hours=hours_back)
        url = f"{self.finnhub_base}/news"
        params = {'category': 'general', 'token': self.finnhub_api_key}

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                print(f"  Error: Status {response.status_code}")
                return []

            data = response.json()
            articles = []
            for item in data:
                pub_time = self._naive(datetime.fromtimestamp(item['datetime']))
                if pub_time > reference_time:
                    continue
                if pub_time < cutoff_time:
                    continue

                text = self._fetch_article_text(item['url'], item['headline'])
                if not text or len(text) < 100:
                    text = f"{item['headline']}. {item.get('summary', '')}"

                articles.append({
                    'title': item['headline'],
                    'url': item['url'],
                    'published': pub_time,
                    'text': text,
                    'summary': item.get('summary', ''),
                    'source': 'finnhub_market'
                })

                if len(articles) >= self.max_market_news:
                    break

            print(f"  Found {len(articles)} market news articles")
            return articles

        except Exception as e:
            print(f"  Error fetching Finnhub market news: {e}")
            return []

    def fetch_google_news(self, symbol: str, hours_back: int = 24, reference_time: datetime = None) -> List[Dict]:
        print(f"Fetching Google News for {symbol}...")
        if reference_time is None:
            reference_time = datetime.now()
        reference_time = self._naive(reference_time)

        cutoff_time = reference_time - timedelta(hours=hours_back)
        query = f"{symbol} stock"
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"

        try:
            feed = feedparser.parse(url)
            articles = []

            for entry in feed.entries[:self.max_google_news * 2]:
                try:
                    pub_time = self._naive(datetime(*entry.published_parsed[:6]))
                except Exception:
                    pub_time = reference_time

                if pub_time > reference_time:
                    continue
                if pub_time < cutoff_time:
                    continue

                title = entry.title
                link = entry.link
                text = self._fetch_article_text(link, title)

                if text:
                    articles.append({
                        'title': title,
                        'url': link,
                        'published': pub_time,
                        'text': text,
                        'source': 'google_news'
                    })

                if len(articles) >= self.max_google_news:
                    break

                time.sleep(0.5)

            print(f"  Found {len(articles)} Google News articles")
            return articles

        except Exception as e:
            print(f"  Error fetching Google News: {e}")
            return []

    def _fetch_article_text(self, url: str, title: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.split()[:1000]
            text = ' '.join(words)
            full_text = f"{title}. {text}"
            return full_text if len(full_text) > 100 else ""

        except Exception:
            return title

    def analyze_sentiment_finbert(self, text: str) -> float:
        import torch
        inputs = self._finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._finbert(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
        return float(probs[0] - probs[1])

    def calculate_recency_weight(self, timestamp: datetime, current_time: datetime) -> float:
        hours_ago = (current_time - timestamp).total_seconds() / 3600
        return float(np.exp(-self.decay_lambda * hours_ago))

    def analyze_symbol(
        self,
        symbol: str,
        hours_back: int = 24,
        reference_time: datetime = None
    ) -> Tuple[float, List[Dict]]:
        if reference_time is None:
            reference_time = datetime.now()
        reference_time = self._naive(reference_time)

        self._load_models()
        current_time = reference_time

        print(f"\n{'='*70}")
        print(f"MEDIA SENTIMENT ANALYSIS FOR {symbol} (reference: {reference_time})")
        print(f"{'='*70}\n")

        company_news = self.fetch_finnhub_company_news(symbol, hours_back, reference_time)
        market_news = self.fetch_finnhub_market_news(hours_back, reference_time)
        google_news = self.fetch_google_news(symbol, hours_back, reference_time)

        if not company_news and not market_news and not google_news:
            print("\nNo recent content found!")
            return 0.5, []

        print(f"\n{'='*70}")
        print(f"ANALYZING SENTIMENT WITH FinBERT")
        print(f"{'='*70}\n")

        results = []
        company_sentiments, company_weights = [], []
        market_sentiments, market_weights = [], []
        google_sentiments, google_weights = [], []

        for i, item in enumerate(company_news):
            print(f"[Company News {i+1}/{len(company_news)}] {item['title'][:60]}...")
            sentiment = self.analyze_sentiment_finbert(item['text'])
            weight = self.calculate_recency_weight(item['published'], current_time)
            results.append({
                'index': len(results) + 1,
                'title': item['title'],
                'url': item['url'],
                'source': 'Finnhub Company News',
                'model': 'FinBERT',
                'sentiment': sentiment,
                'weight': weight,
                'timestamp': item['published']
            })
            company_sentiments.append(sentiment)
            company_weights.append(weight)

        for i, item in enumerate(market_news):
            print(f"[Market News {i+1}/{len(market_news)}] {item['title'][:60]}...")
            sentiment = self.analyze_sentiment_finbert(item['text'])
            weight = self.calculate_recency_weight(item['published'], current_time)
            results.append({
                'index': len(results) + 1,
                'title': item['title'],
                'url': item['url'],
                'source': 'Finnhub Market News',
                'model': 'FinBERT',
                'sentiment': sentiment,
                'weight': weight,
                'timestamp': item['published']
            })
            market_sentiments.append(sentiment)
            market_weights.append(weight)

        for i, item in enumerate(google_news):
            print(f"[Google News {i+1}/{len(google_news)}] {item['title'][:60]}...")
            sentiment = self.analyze_sentiment_finbert(item['text'])
            weight = self.calculate_recency_weight(item['published'], current_time)
            results.append({
                'index': len(results) + 1,
                'title': item['title'],
                'url': item['url'],
                'source': 'Google News',
                'model': 'FinBERT',
                'sentiment': sentiment,
                'weight': weight,
                'timestamp': item['published']
            })
            google_sentiments.append(sentiment)
            google_weights.append(weight)

        weighted_avg = 0.0
        if company_sentiments:
            weighted_avg += self.company_news_weight * np.average(company_sentiments, weights=company_weights)
        if market_sentiments:
            weighted_avg += self.market_news_weight * np.average(market_sentiments, weights=market_weights)
        if google_sentiments:
            weighted_avg += self.google_news_weight * np.average(google_sentiments, weights=google_weights)

        total_weight = (
            (self.company_news_weight if company_sentiments else 0) +
            (self.market_news_weight if market_sentiments else 0) +
            (self.google_news_weight if google_sentiments else 0)
        )
        if total_weight > 0:
            weighted_avg /= total_weight

        confidence = (weighted_avg + 1) / 2
        return float(confidence), results
