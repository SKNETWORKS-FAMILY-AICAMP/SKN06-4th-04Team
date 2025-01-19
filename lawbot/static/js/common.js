function deleteChats() {
    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML = '';
}
function introduceChatbot() {
    const chatBox = document.getElementById("chat-box");

    const introChat = document.createElement("div");
    introChat.classList.add("intro-chat", "chat-message");

    const chatInfo = document.createElement("div");
    chatInfo.classList.add("chat-info");

    const img = document.createElement("img");
    img.classList.add("img-bot");
    img.src = "/static/img/icon-bot.svg";
    chatInfo.appendChild(img);

    const chatBubble = document.createElement("div");
    chatBubble.classList.add("chat-bubble");
    chatBubble.innerHTML = "안녕하세요!<b> 세법 챗봇</b> 'Lawbot' 입니다!<br>무엇을 도와드릴까요?";

    const chatChip = document.createElement("div");
    chatChip.classList.add("chat-chip");

    const chips = ["연말정산", "부가가치세", "근로소득세", "사업소득세", "법인세"];
    chips.forEach(chip => {
        const button = document.createElement("button");
        button.textContent = chip;
        chatChip.appendChild(button);
    });

    introChat.appendChild(chatInfo);
    introChat.appendChild(chatBubble);
    introChat.appendChild(chatChip);

    chatBox.appendChild(introChat);
};