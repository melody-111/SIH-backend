import datetime
import uuid
import re
from typing import Dict, Any, List, Optional, Tuple

from app.db.database import DatabaseManager

class ConversationalMemoryService:
    """
    Privacy-conscious conversational memory and user preferences store.
    Backed by persistent DatabaseManager with SQLite ACID transactions
    and in-memory context caching.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConversationalMemoryService, cls).__new__(cls)
            cls._instance.db = DatabaseManager()
            # conversation_id -> list of message turns (fast context window cache)
            cls._instance._conversations: Dict[str, List[Dict[str, Any]]] = {}
            cls._instance._max_turns_per_conversation = 15
        return cls._instance

    # -------------------------------------------------------------------------
    # A. Current Conversation Context
    # -------------------------------------------------------------------------
    def add_turn(
        self,
        conversation_id: str,
        role: str,
        text: str,
        intent: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location_name: Optional[str] = None,
        target_time: Optional[str] = None
    ):
        """Records a conversational turn both in-memory and persistently in DB."""
        if not conversation_id:
            return

        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []

        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        turn = {
            "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
            "role": role,
            "text": text,
            "intent": intent,
            "latitude": latitude,
            "longitude": longitude,
            "location_name": location_name,
            "target_time": target_time,
            "timestamp": now_iso
        }
        self._conversations[conversation_id].append(turn)

        # Enforce context window limit
        if len(self._conversations[conversation_id]) > self._max_turns_per_conversation:
            self._conversations[conversation_id] = self._conversations[conversation_id][-self._max_turns_per_conversation:]

        # Persist to database
        try:
            self.db.add_conversation_turn(
                conversation_id=conversation_id,
                role=role,
                text=text,
                intent=intent,
                latitude=latitude,
                longitude=longitude,
                location_name=location_name,
                target_time=target_time,
                max_turns=self._max_turns_per_conversation
            )
        except Exception:
            pass

    def get_conversation_context(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Returns recent turns for a conversation from memory or DB."""
        if conversation_id in self._conversations and self._conversations[conversation_id]:
            return self._conversations[conversation_id]

        try:
            db_turns = self.db.get_conversation_turns(conversation_id)
            if db_turns:
                self._conversations[conversation_id] = db_turns
                return db_turns
        except Exception:
            pass

        return []

    def clear_conversation(self, conversation_id: str):
        """Clears current conversation context from memory and database."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
        try:
            self.db.clear_conversation(conversation_id)
        except Exception:
            pass

    def resolve_followup_reference(
        self,
        conversation_id: str,
        query_text: str,
        current_lat: Optional[float] = None,
        current_lon: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Resolves conversational follow-ups such as:
        - "what about tomorrow?" / "kal kaisa rahega?"
        - "compare it with yesterday"
        - "what about there?" / "wahan kaisa hai?"
        """
        turns = self.get_conversation_context(conversation_id)
        msg_lower = (query_text or "").lower().strip()
        now = datetime.datetime.now(datetime.timezone.utc)

        # Find most recent prior location and subject
        last_lat = current_lat
        last_lon = current_lon
        last_loc_name = None
        last_intent = None

        for turn in reversed(turns):
            if turn.get("latitude") is not None and turn.get("longitude") is not None:
                if last_lat is None or last_lat == 0.0:
                    last_lat = turn["latitude"]
                    last_lon = turn["longitude"]
                    last_loc_name = turn.get("location_name")
            if turn.get("intent"):
                if not last_intent:
                    last_intent = turn["intent"]

        resolved_lat = last_lat if (last_lat is not None and last_lat != 0.0) else current_lat
        resolved_lon = last_lon if (last_lon is not None and last_lon != 0.0) else current_lon
        resolved_time = now
        resolved_time_str = now.isoformat()
        is_followup = False

        # Temporal follow-ups
        if any(w in msg_lower for w in ["tomorrow", "kal", "agla din"]):
            resolved_time = now + datetime.timedelta(days=1)
            resolved_time_str = resolved_time.isoformat()
            is_followup = True
        elif any(w in msg_lower for w in ["day after tomorrow", "parso", "parson"]):
            resolved_time = now + datetime.timedelta(days=2)
            resolved_time_str = resolved_time.isoformat()
            is_followup = True
        elif any(w in msg_lower for w in ["yesterday", "beeta kal", "kal pichhla"]):
            resolved_time = now - datetime.timedelta(days=1)
            resolved_time_str = resolved_time.isoformat()
            is_followup = True

        # Spatial follow-ups
        if any(w in msg_lower for w in ["there", "wahan", "udhar", "same place", "usi jagah"]):
            is_followup = True

        # Ambiguity & Freshness check
        needs_clarification = False
        clarification_prompt = None
        is_stale_session = False

        if turns:
            latest_turn_time_str = turns[-1].get("timestamp")
            if latest_turn_time_str:
                try:
                    latest_turn_time = datetime.datetime.fromisoformat(latest_turn_time_str.replace("Z", "+00:00"))
                    if (now - latest_turn_time).total_seconds() > 7200:  # Older than 2 hours
                        is_stale_session = True
                except Exception:
                    pass

        if is_followup:
            if resolved_lat is None or resolved_lon is None:
                needs_clarification = True
                clarification_prompt = "Which coastal location or harbor should I check this for?"
            elif is_stale_session and any(w in msg_lower for w in ["there", "wahan", "udhar"]):
                needs_clarification = True
                clarification_prompt = f"Previous conversation is over 2 hours old. Would you like me to check {last_loc_name or 'the earlier location'} or your current position?"

        return {
            "is_followup": is_followup,
            "needs_clarification": needs_clarification,
            "clarification_prompt": clarification_prompt,
            "is_stale_session": is_stale_session,
            "resolved_latitude": resolved_lat,
            "resolved_longitude": resolved_lon,
            "resolved_location_name": last_loc_name,
            "resolved_time_iso": resolved_time_str,
            "inherited_intent": last_intent
        }

    def get_llm_chat_history(self, conversation_id: str, limit: int = 6) -> List[Dict[str, str]]:
        """
        Formats recent turns for LLM chat completion context.
        Maps internal roles ('user', 'assistant') to standard chat roles.
        """
        turns = self.get_conversation_context(conversation_id)
        if not turns:
            return []
        recent = turns[-limit:]
        return [
            {
                "role": "user" if t.get("role", "user").lower() in ["user", "human"] else "assistant",
                "content": t.get("text", "")
            }
            for t in recent
            if t.get("text")
        ]

    def detect_and_apply_user_corrections(
        self,
        conversation_id: str,
        message: str
    ) -> Optional[Dict[str, Any]]:
        """
        Detects explicit user corrections such as:
        - 'Actually I am in Porbandar, not Goa'
        - 'Use knots instead of km/h'
        - 'I meant tomorrow, not today'
        Returns updated parameters if correction is detected.
        """
        msg_lower = (message or "").lower()
        if not any(indicator in msg_lower for indicator in ["actually", "i meant", "not ", "instead of", "galat", "sahi nahi", "nahi main"]):
            return None

        correction = {}
        # Unit correction
        if "knots" in msg_lower:
            correction["unit_override"] = {"wind_speed": "knots"}
        elif "km/h" in msg_lower or "kmph" in msg_lower:
            correction["unit_override"] = {"wind_speed": "km/h"}

        # Temporal correction
        if "tomorrow" in msg_lower or "kal" in msg_lower:
            now = datetime.datetime.now(datetime.timezone.utc)
            correction["time_override"] = (now + datetime.timedelta(days=1)).isoformat()
        elif "today" in msg_lower or "aaj" in msg_lower:
            now = datetime.datetime.now(datetime.timezone.utc)
            correction["time_override"] = now.isoformat()

        return correction if correction else None

    def summarize_context_if_needed(self, conversation_id: str) -> Optional[str]:
        """
        Summarizes older dialogue turns when history grows beyond 10 turns
        to prevent context window overflow while preserving key decisions.
        """
        turns = self.get_conversation_context(conversation_id)
        if len(turns) <= 10:
            return None

        older_turns = turns[:-6]
        topics = set()
        locations = set()
        for t in older_turns:
            if t.get("intent"):
                topics.add(t["intent"])
            if t.get("location_name"):
                locations.add(t["location_name"])

        loc_str = ", ".join(locations) if locations else "coastal waters"
        top_str = ", ".join(topics) if topics else "general inquiries"
        return f"Prior conversation summary: User explored {top_str} around {loc_str}."

    # -------------------------------------------------------------------------
    # B. User Preferences
    # -------------------------------------------------------------------------
    def get_preferences(self, user_id: str) -> Dict[str, Any]:
        """Returns preferences for user with database persistence."""
        try:
            return self.db.get_user_preferences(user_id)
        except Exception:
            return {
                "user_id": user_id,
                "preferred_language": "en",
                "preferred_units": {
                    "wind_speed": "km/h",
                    "wave_height": "m",
                    "temperature": "°C",
                    "pressure": "hPa"
                },
                "saved_locations": [],
                "enable_spoken_responses": True,
                "spoken_speed": 1.0
            }

    def update_preferences(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Updates user preferences in database after validating schemas.
        Rejects invalid languages or unphysical unit representations.
        """
        # Validate preferred_language if provided
        if "preferred_language" in updates:
            lang = str(updates["preferred_language"]).lower().strip()
            allowed_langs = {"en", "hi", "ta", "te", "bn", "mr", "gu", "ml", "kn", "or"}
            if lang not in allowed_langs:
                raise ValueError(f"Unsupported language code '{lang}'. Must be one of: {sorted(list(allowed_langs))}")

        # Validate preferred_units if provided
        if "preferred_units" in updates and isinstance(updates["preferred_units"], dict):
            units = updates["preferred_units"]
            if "wind_speed" in units and units["wind_speed"].lower() not in {"km/h", "knots", "m/s", "mph"}:
                raise ValueError(f"Invalid wind_speed unit '{units['wind_speed']}'. Allowed: km/h, knots, m/s, mph")
            if "wave_height" in units and units["wave_height"].lower() not in {"m", "ft", "meters"}:
                raise ValueError(f"Invalid wave_height unit '{units['wave_height']}'. Allowed: m, ft")
            if "temperature" in units and units["temperature"].lower() not in {"celsius", "°c", "fahrenheit", "°f"}:
                raise ValueError(f"Invalid temperature unit '{units['temperature']}'. Allowed: celsius, fahrenheit")

        try:
            return self.db.update_user_preferences(user_id, updates)
        except ValueError:
            raise
        except Exception:
            current = self.get_preferences(user_id)
            current.update(updates)
            return current

    # -------------------------------------------------------------------------
    # C. User-Approved Long-Term Memories (Explicit CRUD)
    # -------------------------------------------------------------------------
    def list_saved_memories(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Lists user-approved saved memories with tenant isolation and pagination."""
        try:
            return self.db.list_user_memories(user_id, limit=limit, offset=offset)
        except Exception:
            return []

    def save_memory(
        self,
        user_id: str,
        title: str,
        content: str,
        category: str = "PREFERENCE"
    ) -> Dict[str, Any]:
        """
        Saves user-approved explicit memory.
        Enforces privacy boundary: rejects API keys, passwords, or credentials.
        """
        sensitive_patterns = [
            r"password", r"api_key", r"token", r"secret", r"bearer", r"credential"
        ]
        text_to_check = f"{title} {content}".lower()
        for pat in sensitive_patterns:
            if re.search(pat, text_to_check):
                raise ValueError("Privacy Policy Violation: Storing passwords, API keys, or credentials in memory is prohibited.")

        mem_id = f"mem_{uuid.uuid4().hex[:10]}"
        return self.db.save_user_memory(
            user_id=user_id,
            memory_id=mem_id,
            title=title.strip(),
            content=content.strip(),
            category=category.upper()
        )

    def update_memory(
        self,
        user_id: str,
        memory_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Updates an existing user memory."""
        return self.db.update_user_memory(
            user_id=user_id,
            memory_id=memory_id,
            title=title.strip() if title else None,
            content=content.strip() if content else None
        )

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Deletes a saved user memory."""
        return self.db.delete_user_memory(user_id, memory_id)

    # -------------------------------------------------------------------------
    # D. Personalization Feedback & Governance
    # -------------------------------------------------------------------------
    def record_feedback(
        self,
        user_id: str,
        message_id: str,
        rating: int,  # 1 to 5
        feedback_notes: Optional[str] = None,
        consent_given: bool = True,
        correlation_id: Optional[str] = None,
        category: str = "GENERAL"
    ) -> Dict[str, Any]:
        """Records user feedback under privacy-governance rules and persists to SQLite."""
        if not consent_given:
            return {"status": "SKIPPED", "message": "Feedback skipped without explicit consent."}

        # Privacy Policy Enforcement: Reject any passwords, tokens, or credentials
        if feedback_notes:
            sensitive_patterns = [r"password", r"api_key", r"token", r"secret", r"bearer", r"credential"]
            text_to_check = feedback_notes.lower()
            for pat in sensitive_patterns:
                if re.search(pat, text_to_check):
                    raise ValueError("Privacy Policy Violation: Feedback cannot contain credentials, API keys, or passwords.")

        feed_id = f"fb_{uuid.uuid4().hex[:8]}"
        saved_record = self.db.save_user_feedback(
            feedback_id=feed_id,
            user_id=user_id,
            message_id=message_id,
            rating=rating,
            correlation_id=correlation_id,
            category=category,
            feedback_notes=feedback_notes,
            consent_given=consent_given
        )

        return {
            "status": "RECORDED",
            "feedback": {
                "feedback_id": saved_record["feedback_id"],
                "user_id": saved_record["user_id"],
                "correlation_id": saved_record["correlation_id"],
                "message_id": saved_record["message_id"],
                "rating": saved_record["rating"],
                "category": saved_record["category"],
                "notes": saved_record["feedback_notes"],
                "consent_given": saved_record["consent_given"],
                "created_at": saved_record["created_at"]
            }
        }

    def list_user_feedback(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Lists feedback submissions for authenticated user with tenant isolation."""
        return self.db.list_user_feedback(user_id=user_id, limit=limit, offset=offset)

    def delete_user_feedback(self, user_id: str, feedback_id: str) -> bool:
        """Deletes feedback owned by user."""
        return self.db.delete_user_feedback(feedback_id=feedback_id, user_id=user_id)

    def export_anonymized_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Exports sanitized feedback examples for offline benchmark quality review."""
        return self.db.export_anonymized_feedback_for_eval(limit=limit)

    def clear_all_for_test(self):
        """Reset conversational memory stores before each test for deterministic test isolation."""
        self._conversations.clear()
        try:
            with self.db.transaction() as conn:
                conn.execute("DELETE FROM conversation_turns;")
                conn.execute("DELETE FROM conversations;")
                conn.execute("DELETE FROM user_memories;")
                conn.execute("DELETE FROM user_preferences;")
                conn.execute("DELETE FROM saved_locations;")
                conn.execute("DELETE FROM user_feedback;")
        except Exception:
            pass
