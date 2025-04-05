function ChatHistory({ chats }) {
  // Sort chats by creation time, newest first
  const orderedChats = [...chats].sort((a, b) => {
    return b.createdAt - a.createdAt;
  });

  return (
    <div className="chat-history">
      {orderedChats.map((chat) => {
        const hasPersian = /[\u0600-\u06FF]/.test(chat.name);
        const textDir = hasPersian ? "rtl" : "ltr";

        return (
          <div key={chat.id} className="chat-item" dir={textDir}>
            <button
              className="chat-button"
              dir={textDir}
              onClick={() => handleChatSelect(chat.id)}
              style={{
                borderLeft: hasPersian ? "none" : undefined,
                borderRight: hasPersian ? "none" : undefined,
              }}
            >
              <span className="chat-name-wrapper">{chat.name}</span>
            </button>
            {}
          </div>
        );
      })}
    </div>
  );
}
