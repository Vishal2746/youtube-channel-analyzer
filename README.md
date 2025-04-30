# YouTube Channel Analyzer

A powerful tool for analyzing YouTube channels using YouTube Data API, sentiment analysis, and Google's Gemini AI for enhanced insights.

## Features

- Channel performance analysis
- Multi-channel comparison
- Sentiment analysis of comments
- AI-powered content insights using Google's Gemini
- Detailed PDF report generation
- Interactive visualizations
- Date range filtering
- Customizable analysis options

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Vishal2746/youtube-channel-analyzer.git
cd youtube-channel-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with:
```env
YOUTUBE_API_KEY=your_youtube_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Enter YouTube channel URLs (one per line)
3. Select date range and analysis options
4. Click "Analyze" to start the analysis

## Analysis Types

1. **Channel Summary**: Basic metrics and AI-powered analysis
2. **Comment Analysis**: Sentiment analysis of comments
3. **Complete Analysis**: Full analysis including both metrics and sentiment

## Requirements

- Python 3.8+
- YouTube Data API v3 key
- Google Gemini API key
- Required Python packages (see requirements.txt)

## Deployment

The app can be deployed on Streamlit Cloud:

1. Push code to GitHub
2. Connect your GitHub repo to Streamlit Cloud
3. Add environment variables in Streamlit Cloud settings
4. Deploy!

## File Structure

```
youtube-channel-analyzer/
├── app.py                     # Main Streamlit application
├── youtube_analyzer.py        # YouTube data fetching and analysis
├── ml_model.py               # ML model for sentiment analysis
├── detailed_report_generator.py # PDF report generation
├── cache_manager.py          # Caching system
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (local only)
└── README.md                 # Project documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use this project for any purpose.

## Acknowledgments

- YouTube Data API
- Google Gemini AI
- Hugging Face Transformers
- Streamlit
