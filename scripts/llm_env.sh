# usage: source scripts/llm_env.sh      (zsh)
export LLM_BACKEND=openai
export LLM_BASE_URL="https://api.groq.com/openai/v1"
export LLM_MODEL="openai/gpt-oss-20b"
if [ -z "$LLM_API_KEY" ]; then
  read -s "LLM_API_KEY?Paste your Groq API key: "
  echo
  export LLM_API_KEY
fi