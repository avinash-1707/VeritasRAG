import uuid

from django.db import models

from apps.documents.models import Chunk
from apps.users.models import CustomUser


class ChatSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='chat_sessions')
    title = models.CharField(max_length=500)
    document_ids = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'chat_sessions'
        ordering = ['-updated_at']

    def __str__(self) -> str:
        return self.title


class Message(models.Model):
    ROLE_USER = 'user'
    ROLE_ASSISTANT = 'assistant'
    ROLE_CHOICES = [(ROLE_USER, 'User'), (ROLE_ASSISTANT, 'Assistant')]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    retrieval_score = models.FloatField(null=True, blank=True)
    grounding_score = models.FloatField(null=True, blank=True)
    latency_ms = models.IntegerField(null=True, blank=True)
    cache_hit = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'messages'
        ordering = ['created_at']

    def __str__(self) -> str:
        return f'{self.role}: {self.content[:60]}'


class Citation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name='citations')
    chunk = models.ForeignKey(Chunk, on_delete=models.CASCADE, related_name='citations')
    similarity_score = models.FloatField()
    citation_order = models.IntegerField()

    class Meta:
        db_table = 'citations'
        ordering = ['citation_order']

    def __str__(self) -> str:
        return f'Citation {self.citation_order} for message {self.message_id}'


class QueryLog(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, related_name='query_logs')
    question_text = models.TextField()
    doc_ids = models.JSONField(default=list)
    latency_ms = models.IntegerField(null=True, blank=True)
    chunk_count = models.IntegerField(default=0)
    top_similarity_score = models.FloatField(null=True, blank=True)
    grounding_score = models.FloatField(null=True, blank=True)
    cache_hit = models.BooleanField(default=False)
    model_used = models.CharField(max_length=100, default='google/gemini-3.1-flash-lite')
    low_confidence = models.BooleanField(default=False, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'query_logs'
        ordering = ['-created_at']

    def __str__(self) -> str:
        return f'Query: {self.question_text[:60]}'
