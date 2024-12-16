To create and run the project:

1. Create the directory structure:
```bash
mkdir health_claims_system
cd health_claims_system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Create the files as shown above

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Create .env file:
```env
OPENAI_API_KEY=your_key_here
```

5. Run the application:
```bash
# Terminal - UI
streamlit run ui/app.py
```
