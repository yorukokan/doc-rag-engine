class ConversationMemory:
    """Son N sohbet turunu kayan pencere (sliding window) ile saklar."""

    def __init__(self, max_turns: int = 5) -> None:
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns en az 1 olan bir tam sayı olmalıdır.")
        self.max_turns = max_turns
        self._messages: list[dict[str, str]] = []

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self._messages.append({"role": "user", "content": user_message})
        self._messages.append({"role": "assistant", "content": assistant_message})
        max_messages = self.max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    def get_messages(self) -> list[dict[str, str]]:
        return [dict(msg) for msg in self._messages]

    def clear(self) -> None:
        self._messages.clear()

    def is_empty(self) -> bool:
        return len(self._messages) == 0

    @property
    def turn_count(self) -> int:
        return len(self._messages) // 2

    def __len__(self) -> int:
        return self.turn_count
