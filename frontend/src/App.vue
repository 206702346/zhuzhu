<template>
  <div class="page">
    <header class="hero">
      <div class="badge">RAG Course QA</div>
      <h1>基于 RAG 的课程知识库问答系统</h1>
      <p>上传文档，自动切分、向量检索，并基于知识库回答问题。</p>
    </header>

    <main class="container">
      <section class="card">
        <div class="card-head">
          <h2>1. 知识库管理</h2>
          <button class="btn secondary" @click="refreshStats" :disabled="loadingStats">
            {{ loadingStats ? '刷新中...' : '刷新状态' }}
          </button>
        </div>

        <div v-if="stats" class="stats">
          <div class="stat">
            <span class="label">文档数</span>
            <span class="value">{{ stats.document_count }}</span>
          </div>
          <div class="stat">
            <span class="label">Chunk 数</span>
            <span class="value">{{ stats.chunk_count }}</span>
          </div>
          <div class="stat">
            <span class="label">向量数</span>
            <span class="value">{{ stats.vector_count }}</span>
          </div>
        </div>

        <div class="file-row">
          <input type="file" accept=".txt,.md,.pdf" @change="onFileChange" />
          <button class="btn" @click="uploadDocument" :disabled="!selectedFile || uploading">
            {{ uploading ? '上传中...' : '上传并建立索引' }}
          </button>
          <button class="btn danger" @click="clearKnowledgeBase" :disabled="clearing">
            {{ clearing ? '清理中...' : '清空知识库' }}
          </button>
        </div>

        <p class="tip">建议先上传 .txt 或 .md 测试，稳定后再上传 PDF。</p>

        <p v-if="uploadMsg" class="msg success">{{ uploadMsg }}</p>
        <p v-if="errorMsg" class="msg error">{{ errorMsg }}</p>
      </section>

      <section class="card">
        <h2>2. 提问</h2>
        <textarea
          v-model="question"
          rows="6"
          placeholder="例如：进程和线程有什么区别？"
        ></textarea>

        <div class="query-row">
          <label>
            Top K
            <input type="number" v-model.number="topK" min="1" max="10" />
          </label>

          <button class="btn" @click="askQuestion" :disabled="loadingAsk || !question.trim()">
            {{ loadingAsk ? '生成回答中...' : '开始问答' }}
          </button>

          <button class="btn secondary" @click="resetResult">清空结果</button>
        </div>
      </section>

      <section v-if="answer" ref="answerSection" class="card wide">
        <h2>3. 回答结果</h2>
        <div class="markdown-body" v-html="renderedAnswer"></div>

        <h3>引用来源</h3>
        <div v-if="sources.length" class="source-list">
          <article v-for="(item, index) in sources" :key="index" class="source-item">
            <div class="source-meta">
              <span>[引用 {{ index + 1 }}]</span>
              <span>{{ item.source }}</span>
              <span>chunk {{ item.chunk_id }}</span>
              <span>最终分 {{ formatScore(item.score) }}</span>
              <span v-if="item.hybrid_score != null">融合分 {{ formatScore(item.hybrid_score) }}</span>
              <span v-if="item.rerank_score != null">重排分 {{ formatScore(item.rerank_score) }}</span>
              <span v-if="item.lexical_score != null">词面分 {{ formatScore(item.lexical_score) }}</span>
            </div>
            <div class="source-text">{{ item.text }}</div>

            <div class="score-breakdown">
              <span v-if="item.vector_score != null">向量 {{ formatScore(item.vector_score) }}</span>
              <span v-if="item.bm25_score != null">BM25 {{ formatScore(item.bm25_score) }}</span>
              <span v-if="item.match_sources?.length">命中: {{ item.match_sources.join(' + ') }}</span>
            </div>

          </article>
        </div>
        <p v-else class="tip">本次没有返回引用来源。</p>
      </section>

      <section class="card wide">
        <div class="card-head">
          <h2>4. 历史记录</h2>
          <button class="btn secondary" @click="clearHistory" :disabled="!historyList.length">
            清空历史
          </button>
        </div>

        <p v-if="!historyList.length" class="tip">暂无历史记录，先去问几个问题吧。</p>

        <div v-else class="history-list">
          <article v-for="item in historyList" :key="item.id" class="history-item"
            :class="{ active: item.id === selectedHistoryId }">
            <div class="history-meta">
              <span class="history-time">{{ item.createdAt }}</span>
              <span class="history-topk">TopK: {{ item.topK }}</span>
              <span class="history-topk">引用数: {{ item.sources.length }}</span>
            </div>

            <div class="history-question">
              {{ item.question }}
            </div>

            <div class="history-actions">
              <button class="btn" @click="useHistoryItem(item)">恢复</button>
              <button class="btn danger" @click="deleteHistoryItem(item.id)">删除</button>
            </div>
          </article>
        </div>
      </section>

    </main>
  </div>
</template>

<script setup>

import { ref, onMounted, computed, nextTick } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

marked.setOptions({
  breaks: true,
  gfm: true
})

const answerSection = ref(null)

const renderedAnswer = computed(() => {
  if (!answer.value) return ''
  return DOMPurify.sanitize(marked.parse(answer.value))
})

const apiBase = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

const selectedFile = ref(null)
const uploading = ref(false)
const clearing = ref(false)
const uploadMsg = ref('')
const errorMsg = ref('')
const loadingAsk = ref(false)
const loadingStats = ref(false)

const question = ref('')
const topK = ref(5)

const answer = ref('')
const sources = ref([])
const stats = ref(null)

const historyList = ref([])
const selectedHistoryId = ref('')
const HISTORY_KEY = 'rag_course_qa_history_v1'

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY)
    historyList.value = raw ? JSON.parse(raw) : []
  } catch (err) {
    historyList.value = []
  }
}

function saveHistory() {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(historyList.value))
}

function addHistoryItem(questionText, answerText, sourceList, topKValue) {
  const item = {
    id: Date.now() + '_' + Math.random().toString(16).slice(2),
    question: questionText,
    answer: answerText,
    sources: sourceList || [],
    topK: topKValue,
    createdAt: new Date().toLocaleString()
  }

  // 最新的放前面
  historyList.value.unshift(item)

  // 可选：限制最多保存 20 条，避免越来越多
  if (historyList.value.length > 20) {
    historyList.value = historyList.value.slice(0, 20)
  }

  saveHistory()
}

function useHistoryItem(item) {
  selectedHistoryId.value = item.id
  question.value = item.question
  answer.value = item.answer
  sources.value = item.sources || []

  nextTick(() => {
    answerSection.value?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  })
}

function deleteHistoryItem(id) {
  historyList.value = historyList.value.filter(item => item.id !== id)
  if (selectedHistoryId.value === id) {
    selectedHistoryId.value = ''
  }
  saveHistory()
}

function clearHistory() {
  historyList.value = []
  selectedHistoryId.value = ''
  localStorage.removeItem(HISTORY_KEY)
}

function onFileChange(event) {
  selectedFile.value = event.target.files?.[0] ?? null
  uploadMsg.value = ''
  errorMsg.value = ''
}

function resetResult() {
  answer.value = ''
  sources.value = []
  errorMsg.value = ''
}

function formatScore(score) {
  const n = Number(score ?? 0)
  return Number.isFinite(n) ? n.toFixed(4) : '0.0000'
}

async function refreshStats() {
  loadingStats.value = true
  try {
    const res = await fetch(`${apiBase}/documents/stats`)
    if (!res.ok) throw new Error(await res.text())
    stats.value = await res.json()
  } catch (err) {
    stats.value = null
  } finally {
    loadingStats.value = false
  }
}

async function uploadDocument() {
  if (!selectedFile.value) return

  uploading.value = true
  errorMsg.value = ''
  uploadMsg.value = ''

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)

    const res = await fetch(`${apiBase}/documents/upload`, {
      method: 'POST',
      body: formData
    })

    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || '上传失败')
    }

    const data = await res.json()
    uploadMsg.value = `上传成功：${data.filename}，切分 ${data.chunk_count} 段，索引 ${data.vector_count} 个向量。`
    await refreshStats()
  } catch (err) {
    errorMsg.value = `上传失败：${err.message || err}`
  } finally {
    uploading.value = false
  }
}

async function askQuestion() {
  if (!question.value.trim()) return

  loadingAsk.value = true
  errorMsg.value = ''

  try {
    const res = await fetch(`${apiBase}/chat/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        question: question.value,
        top_k: topK.value
      })
    })

    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || '问答失败')
    }

    const data = await res.json()
    answer.value = data.answer || ''
    sources.value = data.sources || []

    addHistoryItem(question.value, answer.value, sources.value, topK.value)
  } catch (err) {
    errorMsg.value = `问答失败：${err.message || err}`
  } finally {
    loadingAsk.value = false
  }
}

async function clearKnowledgeBase() {
  clearing.value = true
  errorMsg.value = ''
  uploadMsg.value = ''

  try {
    const res = await fetch(`${apiBase}/documents/clear`, {
      method: 'DELETE'
    })

    if (!res.ok) {
      const text = await res.text()
      throw new Error(text || '清空失败')
    }

    const data = await res.json()
    uploadMsg.value = data.message || '知识库已清空'
    selectedFile.value = null
    answer.value = ''
    sources.value = []
    await refreshStats()
  } catch (err) {
    errorMsg.value = `清空失败：${err.message || err}`
  } finally {
    clearing.value = false
  }
}

onMounted(() => {
  refreshStats()
  loadHistory()
})
</script>

<style scoped>
.page {
  min-height: 100vh;
  background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
  color: #e5e7eb;
}

.hero {
  max-width: 1100px;
  margin: 0 auto;
  padding: 48px 20px 24px;
}

.badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(59, 130, 246, 0.15);
  color: #93c5fd;
  font-size: 14px;
  margin-bottom: 14px;
}

.hero h1 {
  margin: 0;
  font-size: 34px;
  line-height: 1.2;
}

.hero p {
  margin: 12px 0 0;
  color: #94a3b8;
}

.container {
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 20px 48px;
  display: grid;
  gap: 20px;
  grid-template-columns: 1fr 1fr;
}

.card {
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 20px;
  padding: 20px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.18);
  backdrop-filter: blur(8px);
}

.wide {
  grid-column: 1 / -1;
}

.card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.card h2,
.card h3 {
  margin: 0 0 16px;
}

.stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}

.stat {
  background: rgba(30, 41, 59, 0.8);
  border: 1px solid rgba(148, 163, 184, 0.14);
  border-radius: 16px;
  padding: 14px;
}

.label {
  display: block;
  color: #94a3b8;
  font-size: 13px;
  margin-bottom: 6px;
}

.value {
  font-size: 24px;
  font-weight: 700;
  color: #f8fafc;
}

.file-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
}

input[type='file'] {
  color: #cbd5e1;
}

textarea,
input[type='number'] {
  width: 100%;
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: #0b1220;
  color: #e5e7eb;
  border-radius: 14px;
  padding: 12px 14px;
  outline: none;
}

textarea:focus,
input[type='number']:focus {
  border-color: #60a5fa;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
}

textarea {
  resize: vertical;
}

.query-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: end;
  margin-top: 14px;
}

.query-row label {
  display: flex;
  flex-direction: column;
  gap: 8px;
  color: #cbd5e1;
  min-width: 120px;
}

.btn {
  border: none;
  border-radius: 14px;
  padding: 12px 18px;
  background: #2563eb;
  color: white;
  cursor: pointer;
  transition: 0.2s;
}

.btn:hover {
  transform: translateY(-1px);
  background: #1d4ed8;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.secondary {
  background: #334155;
}

.secondary:hover {
  background: #475569;
}

.danger {
  background: #dc2626;
}

.danger:hover {
  background: #b91c1c;
}

.tip {
  margin-top: 12px;
  color: #94a3b8;
  font-size: 14px;
}

.msg {
  margin-top: 12px;
  padding: 12px 14px;
  border-radius: 14px;
}

.success {
  background: rgba(16, 185, 129, 0.12);
  color: #6ee7b7;
}

.error {
  background: rgba(239, 68, 68, 0.12);
  color: #fca5a5;
}

.answer {
  white-space: pre-wrap;
  background: #0b1220;
  border: 1px solid rgba(148, 163, 184, 0.16);
  padding: 16px;
  border-radius: 16px;
  line-height: 1.7;
  color: #e5e7eb;
  overflow-x: auto;
}

.source-list {
  display: grid;
  gap: 12px;
  margin-top: 12px;
}

.source-item {
  background: #0b1220;
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 16px;
  padding: 14px;
}

.source-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  color: #93c5fd;
  font-size: 13px;
  margin-bottom: 10px;
}

.source-text {
  color: #e5e7eb;
  white-space: pre-wrap;
  line-height: 1.7;
}

.markdown-body {
  background: #0b1220;
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 16px;
  padding: 16px;
  line-height: 1.8;
  color: #e5e7eb;
  overflow-x: auto;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3),
.markdown-body :deep(h4) {
  color: #f8fafc;
  margin: 16px 0 10px;
  line-height: 1.3;
}

.markdown-body :deep(p) {
  margin: 10px 0;
}

.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  padding-left: 1.5rem;
  margin: 10px 0;
}

.markdown-body :deep(li) {
  margin: 6px 0;
}

.markdown-body :deep(code) {
  background: rgba(148, 163, 184, 0.16);
  padding: 2px 6px;
  border-radius: 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.markdown-body :deep(pre) {
  background: #020617;
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 12px;
  padding: 14px;
  overflow: auto;
}

.markdown-body :deep(pre code) {
  background: transparent;
  padding: 0;
}

@media (max-width: 900px) {
  .container {
    grid-template-columns: 1fr;
  }

  .stats {
    grid-template-columns: 1fr;
  }
}
/* 历史记录 */
.history-list {
  display: grid;
  gap: 12px;
}

.history-item {
  background: #0b1220;
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 16px;
  padding: 14px;
  transition: 0.2s;
}

.history-item.active {
  border-color: #60a5fa;
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.12);
}

.history-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  color: #93c5fd;
  font-size: 13px;
  margin-bottom: 10px;
}

.history-question {
  color: #e5e7eb;
  line-height: 1.7;
  white-space: pre-wrap;
  margin-bottom: 12px;
}

.history-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.score-breakdown {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  margin-top: 10px;
  color: #93c5fd;
  font-size: 13px;
}

</style>