import streamlit as st
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import uuid
import sqlite3
from datetime import datetime


class Database:
    def __init__(self, db_path: str = 'chatbot.db'):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        """Create and return a database connection with foreign keys enabled"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('PRAGMA foreign_keys = ON')
        return conn

    def init_db(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = self.get_connection()
            c = conn.cursor()

            c.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    thread_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS message_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
                );

                CREATE INDEX IF NOT EXISTS idx_thread_id ON message_store(thread_id);
                CREATE INDEX IF NOT EXISTS idx_user_threads ON threads(user_id);
            ''')

            conn.commit()
            print("Database initialized successfully")
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
            raise
        finally:
            conn.close()


class ChatManager:
    def __init__(self, db: Database):
        self.db = db

    def create_new_thread(self, user_id: str, thread_name: Optional[str] = None) -> str:
        """Create a new chat thread"""
        thread_id = str(uuid.uuid4())
        if not thread_name:
            thread_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        try:
            conn = self.db.get_connection()
            c = conn.cursor()
            c.execute('INSERT OR IGNORE INTO users (user_id) VALUES (?)', (user_id,))
            c.execute('''
                INSERT INTO threads (thread_id, user_id, thread_name)
                VALUES (?, ?, ?)
            ''', (thread_id, user_id, thread_name))
            conn.commit()
            return thread_id
        except sqlite3.Error as e:
            st.error(f"Failed to create new chat thread: {e}")
            return None
        finally:
            conn.close()

    def get_thread_messages(self, thread_id: str) -> List:
        """Get messages for a specific thread"""
        try:
            conn = self.db.get_connection()
            c = conn.cursor()
            c.execute('''
                SELECT message_type, content 
                FROM message_store 
                WHERE thread_id = ? 
                ORDER BY created_at
            ''', (thread_id,))

            messages = []
            for msg_type, content in c.fetchall():
                if msg_type == "human":
                    messages.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    messages.append(AIMessage(content=content))
            return messages
        except sqlite3.Error as e:
            st.error(f"Error retrieving messages: {e}")
            return []
        finally:
            conn.close()

    def get_user_threads(self, user_id: str):
        """Get all threads for a user"""
        try:
            conn = self.db.get_connection()
            c = conn.cursor()
            c.execute('''
                SELECT thread_id, thread_name, created_at
                FROM threads
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))
            return c.fetchall()
        except sqlite3.Error as e:
            st.error(f"Error fetching threads: {e}")
            return []
        finally:
            conn.close()

    def save_message(self, thread_id: str, message_type: str, content: str):
        """Save message to a specific thread"""
        try:
            conn = self.db.get_connection()
            c = conn.cursor()
            c.execute('''
                INSERT INTO message_store 
                (thread_id, message_type, content)
                VALUES (?, ?, ?)
            ''', (thread_id, message_type, content))
            conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error saving message: {e}")
        finally:
            conn.close()

    def thread_exists(self, thread_id: str) -> bool:
        """Check if a thread exists"""
        try:
            conn = self.db.get_connection()
            c = conn.cursor()
            c.execute('SELECT 1 FROM threads WHERE thread_id = ?', (thread_id,))
            return c.fetchone() is not None
        except sqlite3.Error as e:
            print(f"Error checking thread existence: {e}")
            return False
        finally:
            conn.close()

    def delete_thread(self, thread_id: str):
        """Delete a thread and its messages"""
        try:
            conn = self.db.get_connection()
            c = conn.cursor()
            c.execute('DELETE FROM message_store WHERE thread_id = ?', (thread_id,))
            c.execute('DELETE FROM threads WHERE thread_id = ?', (thread_id,))
            conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error deleting thread: {e}")
        finally:
            conn.close()

    def clear_chat_history(self, thread_id: str):
        """Clear chat history for a thread"""
        try:
            conn = self.db.get_connection()
            c = conn.cursor()
            c.execute('DELETE FROM message_store WHERE thread_id = ?', (thread_id,))
            conn.commit()
        except sqlite3.Error as e:
            st.error(f"Error clearing chat history: {e}")
        finally:
            conn.close()


class AIAgent:
    def __init__(self, openai_api_key: str, serpapi_key: str):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            openai_api_key=openai_api_key,
            streaming=True
        )

        # Initialize SerpAPI
        search = SerpAPIWrapper(serpapi_api_key=serpapi_key)

        # Create tools list
        self.tools = [
            Tool(
                name="Search",
                func=search.run,
                description="useful for when you need to answer questions about current events or the current state of the world"
            )
        ]

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant that can search the internet."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create agent
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=self.tools)

    def stream_response(self, query: str, chat_history: List) -> Optional[str]:
        """Stream the agent's response"""
        placeholder = st.empty()
        full_response = ""

        try:
            response = self.agent_executor.invoke({
                "input": query,
                "chat_history": chat_history
            })

            if isinstance(response.get("output"), str):
                words = response["output"].split()
                for word in words:
                    full_response += word + " "
                    placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)

                placeholder.markdown(full_response)
                return full_response
            return None

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None


def init_session_state(chat_manager: ChatManager):
    """Initialize session state variables"""
    if 'user_id' not in st.session_state:
        if 'uid' in st.query_params:
            st.session_state.user_id = st.query_params['uid']
        else:
            new_id = str(uuid.uuid4())
            st.session_state.user_id = new_id
            st.query_params['uid'] = new_id

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant that can search the internet.")
        ]

    if 'chat_history_loaded' not in st.session_state:
        st.session_state.chat_history_loaded = False

    try:
        conn = chat_manager.db.get_connection()
        c = conn.cursor()

        c.execute('INSERT OR IGNORE INTO users (user_id) VALUES (?)',
                  (st.session_state.user_id,))
        conn.commit()

        c.execute('''
            SELECT thread_id 
            FROM threads 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (st.session_state.user_id,))

        existing_thread = c.fetchone()

        if existing_thread:
            st.session_state.current_thread = existing_thread[0]
            if not st.session_state.chat_history_loaded:
                st.session_state.messages.extend(
                    chat_manager.get_thread_messages(existing_thread[0])
                )
                st.session_state.chat_history_loaded = True
        else:
            default_thread_id = chat_manager.create_new_thread(
                st.session_state.user_id,
                "Default Chat"
            )
            st.session_state.current_thread = default_thread_id
            st.session_state.chat_history_loaded = True

    except sqlite3.Error as e:
        st.error("Failed to initialize chat. Please try refreshing the page.")
    finally:
        conn.close()


def main():
    st.set_page_config(page_title="AI Chatbot", layout="wide")

    # Initialize database and chat manager
    db = Database()
    chat_manager = ChatManager(db)

    # Initialize session state
    init_session_state(chat_manager)

    # API key handling
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    serpapi_key = st.sidebar.text_input("SerpAPI Key", type="password")

    st.title("🤖 AI Chatbot with Web Search")

    # Display thread selector in sidebar
    thread_selector(chat_manager)

    # Verify current thread exists
    if not hasattr(st.session_state, 'current_thread') or \
            not chat_manager.thread_exists(st.session_state.current_thread):
        init_session_state(chat_manager)
        st.rerun()
        return

    # Display chat messages
    for message in st.session_state.messages[1:]:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.markdown(message.content)

    # Chat input handling
    if prompt := st.chat_input("What would you like to know?"):
        if not openai_api_key or not serpapi_key:
            st.error("Please provide both API keys in the sidebar!")
            return

        with st.chat_message("user"):
            st.markdown(prompt)

        chat_manager.save_message(st.session_state.current_thread, "human", prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))

        with st.chat_message("assistant"):
            agent = AIAgent(openai_api_key, serpapi_key)
            response = agent.stream_response(prompt, st.session_state.messages[:-1])

            if response:
                chat_manager.save_message(st.session_state.current_thread, "ai", response)
                st.session_state.messages.append(AIMessage(content=response))


def thread_selector(chat_manager: ChatManager):
    """Render thread selection sidebar"""
    with st.sidebar:
        st.markdown("### Chat Threads")

        # New thread creation
        col1, col2 = st.columns([4, 1])
        with col1:
            new_thread_name = st.text_input(
                "New chat name",
                placeholder="Enter chat name...",
                label_visibility="collapsed"
            )
        with col2:
            if st.button("➕", help="Create new chat"):
                if not new_thread_name:
                    new_thread_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                thread_id = chat_manager.create_new_thread(
                    st.session_state.user_id,
                    new_thread_name
                )
                if thread_id:
                    st.session_state.current_thread = thread_id
                    st.session_state.messages = [
                        SystemMessage(content="You are a helpful AI assistant that can search the internet.")
                    ]
                    st.rerun()

        # Display existing threads
        threads = chat_manager.get_user_threads(st.session_state.user_id)
        if threads:
            st.markdown("### Your Chats")

            for thread_id, thread_name, created_at in threads:
                cols = st.columns([5, 1, 1])

                with cols[0]:
                    is_current = thread_id == st.session_state.current_thread
                    button_type = "primary" if is_current else "secondary"

                    if st.button(
                            f"💬 {thread_name}",
                            key=f"thread_{thread_id}",
                            type=button_type,
                            use_container_width=True,
                    ):
                        st.session_state.current_thread = thread_id
                        st.session_state.messages = [
                            SystemMessage(content="You are a helpful AI assistant that can search the internet.")
                        ]
                        st.session_state.messages.extend(
                            chat_manager.get_thread_messages(thread_id)
                        )
                        st.rerun()

                with cols[1]:
                    if st.button("🧹", key=f"clear_{thread_id}", help="Clear chat history"):
                        chat_manager.clear_chat_history(thread_id)
                        if thread_id == st.session_state.current_thread:
                            st.session_state.messages = [
                                SystemMessage(content="You are a helpful AI assistant that can search the internet.")
                            ]
                        st.rerun()

                with cols[2]:
                    if st.button("🗑️", key=f"del_{thread_id}", help="Delete this chat"):
                        chat_manager.delete_thread(thread_id)
                        if thread_id == st.session_state.current_thread:
                            st.session_state.current_thread = None
                        st.rerun()


if __name__ == "__main__":
    main()