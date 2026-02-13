"""
Media Sentiment Analyzer

Fetches news from Finnhub API and Google News RSS, analyzes with FinBERT.

Data Sources:
- Finnhub Company News API (requires free API key)
- Finnhub Market News API (requires free API key)
- Google News RSS (free, no API key)

Weighting:
- Finnhub Company News: 50% (most relevant)
- Finnhub Market News: 30% (general context)
- Google News RSS: 20% (additional coverage)

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
import os

FINNHUB_API_KEY = "d67n5f1r01qobepik77gd67n5f1r01qobepik780"


class MediaSentimentAnalyzer:
    """
    Analyzes media sentiment using Finnhub API and Google News.
    """
    
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
        """
        Args:
            finnhub_api_key: Finnhub API key (uses hardcoded if not provided)
            recency_decay_lambda: Decay rate for recency weighting
            max_company_news: Max Finnhub company news articles
            max_market_news: Max Finnhub market news articles
            max_google_news: Max Google News RSS articles
            company_news_weight: Weight for company news (default: 0.50)
            market_news_weight: Weight for market news (default: 0.30)
            google_news_weight: Weight for Google news (default: 0.20)
        """
        # Use provided key, or fall back to hardcoded key
        self.finnhub_api_key = finnhub_api_key or FINNHUB_API_KEY
        
        if not self.finnhub_api_key or self.finnhub_api_key == "PASTE_YOUR_API_KEY_HERE":
            raise ValueError(
                "Finnhub API key not set!\n"
                "Edit sentiment_analyzer.py and paste your API key at line 29.\n"
                "Get a free API key at: https://finnhub.io/register"
            )
        
        self.decay_lambda = recency_decay_lambda
        self.max_company_news = max_company_news
        self.max_market_news = max_market_news
        self.max_google_news = max_google_news
        
        self.company_news_weight = company_news_weight
        self.market_news_weight = market_news_weight
        self.google_news_weight = google_news_weight
        
        # Finnhub base URL
        self.finnhub_base = "https://finnhub.io/api/v1"
        
        # Lazy-load sentiment model
        self._finbert = None
        self._finbert_tokenizer = None
    
    def _load_models(self):
        """Load FinBERT model (lazy initialization)."""
        if self._finbert is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            print("Loading FinBERT...")
            self._finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self._finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            print("FinBERT loaded!\n")
    
    def fetch_finnhub_company_news(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """
        Fetch company-specific news from Finnhub API.
        
        Returns:
            List of dicts with 'title', 'url', 'published', 'text', 'summary'
        """
        print(f"Fetching Finnhub company news for {symbol}...")
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(hours=hours_back)
        
        # API endpoint
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
                # Parse datetime
                pub_time = datetime.fromtimestamp(item['datetime'])
                
                # Try to fetch full article text, fallback to summary
                text = self._fetch_article_text(item['url'], item['headline'])
                if not text or len(text) < 100:
                    # Use headline + summary if article fetch fails
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
    
    def fetch_finnhub_market_news(self, hours_back: int = 24) -> List[Dict]:
        """
        Fetch general market news from Finnhub API.
        
        Returns:
            List of dicts with 'title', 'url', 'published', 'text', 'summary'
        """
        print(f"Fetching Finnhub market news...")
        
        # API endpoint
        url = f"{self.finnhub_base}/news"
        params = {
            'category': 'general',
            'token': self.finnhub_api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"  Error: Status {response.status_code}")
                return []
            
            data = response.json()
            
            articles = []
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for item in data:
                # Parse datetime
                pub_time = datetime.fromtimestamp(item['datetime'])
                
                # Filter by time
                if pub_time < cutoff_time:
                    continue
                
                # Try to fetch full article text, fallback to summary
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
    
    def fetch_google_news(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """
        Fetch news from Google News RSS feed (free, no API key needed).
        
        Returns:
            List of dicts with 'title', 'url', 'published', 'text'
        """
        print(f"Fetching Google News for {symbol}...")
        
        # Google News RSS URL
        query = f"{symbol} stock"
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            # Parse RSS feed
            feed = feedparser.parse(url)
            
            articles = []
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for entry in feed.entries[:self.max_google_news * 2]:  # Get extra in case some are old
                # Parse published time
                try:
                    pub_time = datetime(*entry.published_parsed[:6])
                except:
                    pub_time = datetime.now()  # If can't parse, assume recent
                
                # Filter by time
                if pub_time < cutoff_time:
                    continue
                
                # Get article details
                title = entry.title
                link = entry.link
                
                # Fetch full article text
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
                
                # Rate limit to be polite
                time.sleep(0.5)
            
            print(f"  Found {len(articles)} Google News articles")
            return articles
            
        except Exception as e:
            print(f"  Error fetching Google News: {e}")
            return []
    
    def _fetch_article_text(self, url: str, title: str) -> str:
        """
        Fetch and extract text from a news article URL.
        
        Args:
            url: Article URL
            title: Article title to prepend
            
        Returns:
            Extracted article text (title + first 1000 words)
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Get text from paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            
            # Clean whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Limit to first 1000 words (enough for sentiment)
            words = text.split()[:1000]
            text = ' '.join(words)
            
            # Prepend title for context
            full_text = f"{title}. {text}"
            
            return full_text if len(full_text) > 100 else ""
            
        except Exception as e:
            # If article fetch fails, at least use the title
            return title
    
    def analyze_sentiment_finbert(self, text: str) -> float:
        """
        Analyze with FinBERT.
        
        Returns:
            Sentiment score in [-1, 1]
        """
        import torch
        
        inputs = self._finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self._finbert(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
        
        # FinBERT: [positive, negative, neutral]
        return float(probs[0] - probs[1])
    
    def calculate_recency_weight(self, timestamp: datetime, current_time: datetime) -> float:
        """Calculate exponential decay weight based on age."""
        hours_ago = (current_time - timestamp).total_seconds() / 3600
        weight = np.exp(-self.decay_lambda * hours_ago)
        return float(weight)
    
    def analyze_symbol(
        self,
        symbol: str,
        hours_back: int = 24
    ) -> Tuple[float, List[Dict]]:
        """
        Complete analysis pipeline for a symbol.
        
        Args:
            symbol: Stock ticker
            hours_back: How far back to look for content
            
        Returns:
            (confidence_score, detailed_results)
        """
        # Load models
        self._load_models()
        
        current_time = datetime.now()
        
        # Fetch content
        print(f"\n{'='*70}")
        print(f"MEDIA SENTIMENT ANALYSIS FOR {symbol}")
        print(f"{'='*70}\n")
        
        company_news = self.fetch_finnhub_company_news(symbol, hours_back)
        market_news = self.fetch_finnhub_market_news(hours_back)
        google_news = self.fetch_google_news(symbol, hours_back)
        
        if not company_news and not market_news and not google_news:
            print("\nNo recent content found!")
            return 0.5, []
        
        # Analyze sentiment
        print(f"\n{'='*70}")
        print(f"ANALYZING SENTIMENT WITH FinBERT")
        print(f"{'='*70}\n")
        
        results = []
        company_sentiments = []
        company_weights = []
        market_sentiments = []
        market_weights = []
        google_sentiments = []
        google_weights = []
        
        # Analyze company news
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
        
        # Analyze market news
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
        
        # Analyze Google news
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
        
        # Calculate weighted confidence with source-based weighting
        weighted_avg = 0.0
        
        if company_sentiments:
            company_avg = np.average(company_sentiments, weights=company_weights)
            weighted_avg += self.company_news_weight * company_avg
        
        if market_sentiments:
            market_avg = np.average(market_sentiments, weights=market_weights)
            weighted_avg += self.market_news_weight * market_avg
        
        if google_sentiments:
            google_avg = np.average(google_sentiments, weights=google_weights)
            weighted_avg += self.google_news_weight * google_avg
        
        # Normalize weights if some sources missing
        total_weight = 0.0
        if company_sentiments:
            total_weight += self.company_news_weight
        if market_sentiments:
            total_weight += self.market_news_weight
        if google_sentiments:
            total_weight += self.google_news_weight
        
        if total_weight > 0:
            weighted_avg /= total_weight
        
        # Normalize to [0, 1]
        confidence = (weighted_avg + 1) / 2
        
        return float(confidence), results
