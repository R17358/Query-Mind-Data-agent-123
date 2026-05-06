require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

const dns = require("dns");
dns.setServers(['8.8.8.8', '8.8.4.4']);

// mongodb+srv://Ritesh_Ecom:Ritesh5484@riteshcluster.kqabiaz.mongodb.net/player_app?appName=Riteshcluster

// ── Serve frontend ────────────────────────────────────────────
app.use(express.static(path.join(__dirname, 'public')));
app.get('/agent', (req, res) => res.sendFile(path.join(__dirname, 'index.html')));

// ── AI Router ─────────────────────────────────────────────────
async function callAI(messages, systemPrompt) {
  const provider = process.env.AI_PROVIDER || 'gemini';

  if (provider === 'gemini') {
    const { GoogleGenerativeAI } = require('@google/generative-ai');
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const model = genAI.getGenerativeModel({
      model: process.env.GEMINI_MODEL || 'gemini-2.5-flash',
      systemInstruction: systemPrompt || ''
    });
    // Convert messages array to Gemini format
    const history = messages.slice(0, -1).map(m => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content }]
    }));
    const lastMsg = messages[messages.length - 1].content;
    const chat = model.startChat({ history });
    const result = await chat.sendMessage(lastMsg);
    return result.response.text();
  }

  if (provider === 'openai') {
    const { OpenAI } = require('openai');
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    const res = await openai.chat.completions.create({
      model: process.env.OPENAI_MODEL || 'gpt-4o-mini',
      messages: systemPrompt
        ? [{ role: 'system', content: systemPrompt }, ...messages]
        : messages
    });
    return res.choices[0].message.content;
  }

  if (provider === 'anthropic') {
    const Anthropic = require('@anthropic-ai/sdk');
    const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    const res = await client.messages.create({
      model: process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-20250514',
      max_tokens: 2048,
      system: systemPrompt || '',
      messages
    });
    return res.content[0].text;
  }

  throw new Error('Unknown AI_PROVIDER: ' + provider);
}

// ── Schema Discovery ──────────────────────────────────────────
async function discoverSchema(connectionString, dbType) {
  if (dbType === 'mongodb') {
    const mongoose = require('mongoose');
    const conn = await mongoose.createConnection(connectionString).asPromise();
    const db = conn.db;
    const collections = await db.listCollections().toArray();
    const schema = [];
    for (const col of collections.slice(0, 30)) {
      const samples = await db.collection(col.name).find({}).limit(5).toArray();
      const count = await db.collection(col.name).estimatedDocumentCount();
      const fieldMap = {};
      samples.forEach(doc => {
        Object.entries(doc).forEach(([k, v]) => {
          if (!fieldMap[k]) fieldMap[k] = inferMongoType(v);
        });
      });
      schema.push({ name: col.name, fields: Object.entries(fieldMap).map(([name, type]) => ({ name, type })), count });
    }
    await conn.close();
    return schema;
  }
  if (dbType === 'postgresql') {
    const { Pool } = require('pg');
    const pool = new Pool({ connectionString });
    const { rows } = await pool.query(
      `SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' ORDER BY table_name, ordinal_position`
    );
    const tables = {};
    rows.forEach(r => {
      if (!tables[r.table_name]) tables[r.table_name] = { name: r.table_name, fields: [] };
      tables[r.table_name].fields.push({ name: r.column_name, type: r.data_type });
    });
    await pool.end();
    return Object.values(tables);
  }
  if (dbType === 'mysql') {
    const mysql = require('mysql2/promise');
    const conn = await mysql.createConnection(connectionString);
    const [rows] = await conn.execute(
      `SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = DATABASE() ORDER BY TABLE_NAME, ORDINAL_POSITION`
    );
    const tables = {};
    rows.forEach(r => {
      const t = r.TABLE_NAME || r.table_name;
      if (!tables[t]) tables[t] = { name: t, fields: [] };
      tables[t].fields.push({ name: r.COLUMN_NAME || r.column_name, type: r.DATA_TYPE || r.data_type });
    });
    await conn.end();
    return Object.values(tables);
  }
  throw new Error('Unsupported dbType: ' + dbType);
}

function inferMongoType(value) {
  if (value === null) return 'null';
  const type = typeof value;
  if (type === 'string') return 'String';
  if (type === 'number') return Number.isInteger(value) ? 'Integer' : 'Float';
  if (type === 'boolean') return 'Boolean';
  if (value instanceof Date) return 'Date';
  if (Array.isArray(value)) return 'Array';
  if (type === 'object' && value._bsontype === 'ObjectId') return 'ObjectId';
  if (type === 'object') return 'Object';
  return type;
}

// ── Query Validation & Execution ──────────────────────────────
function validateQuery(query, dbType) {
  const q = query.trim();
  if (dbType === 'mongodb') {
    const blocked = /\.(drop|deleteMany|deleteOne|remove|updateMany|updateOne|insertMany|insertOne|createCollection|dropCollection)\s*\(/i;
    if (blocked.test(q)) throw new Error('Only read operations are allowed');
    return q;
  }
  if (!/^\s*select\b/i.test(q)) throw new Error('Only SELECT queries are allowed for safety');
  return q;
}

async function executeQuery(query, dbType, connectionString, maxResults = 100) {
  const safeQ = validateQuery(query, dbType);
  if (dbType === 'mongodb') {
    const mongoose = require('mongoose');
    const conn = await mongoose.createConnection(connectionString).asPromise();
    const db = conn.db;
    const fn = new Function('db', '"use strict"; return (' + safeQ + ')');
    let result = fn(db);
    if (result && typeof result.then === 'function') result = await result;
    if (result && typeof result.toArray === 'function') result = await result.toArray();
    await conn.close();
    const arr = Array.isArray(result) ? result : [result];
    return arr.slice(0, maxResults);
  }
  if (dbType === 'postgresql') {
    const { Pool } = require('pg');
    const pool = new Pool({ connectionString });
    const limitedQ = /\blimit\b/i.test(safeQ) ? safeQ : safeQ + ' LIMIT ' + maxResults;
    const { rows } = await pool.query(limitedQ);
    await pool.end();
    return rows;
  }
  if (dbType === 'mysql') {
    const mysql = require('mysql2/promise');
    const conn = await mysql.createConnection(connectionString);
    const limitedQ = /\blimit\b/i.test(safeQ) ? safeQ : safeQ + ' LIMIT ' + maxResults;
    const [rows] = await conn.execute(limitedQ);
    await conn.end();
    return rows;
  }
}

// ── Routes ────────────────────────────────────────────────────

// Schema discovery
app.post('/api/schema', async (req, res) => {
  const { connectionString, dbType } = req.body;
  if (!connectionString || !dbType) return res.status(400).json({ error: 'connectionString and dbType are required' });
  try {
    const schema = await discoverSchema(connectionString, dbType);
    res.json({ schema });
  } catch (err) {
    console.error('[schema]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// Natural language DB query — TWO PASS: generate query → run it → AI writes answer from REAL results
app.post('/api/query', async (req, res) => {
  const { question, schema, dbType, connectionString, maxResults } = req.body;
  if (!question || !schema || !dbType || !connectionString) {
    return res.status(400).json({ error: 'question, schema, dbType, connectionString required' });
  }

  const schemaStr = schema.map(t => {
    const name = t.name || t.table;
    const fields = (t.fields || t.columns || []).map(f => typeof f === 'string' ? f : f.name + '(' + f.type + ')');
    return name + ': [' + fields.join(', ') + ']';
  }).join('\n');

  const isMongo = dbType === 'mongodb';

  // ── PASS 1: Generate query only ──────────────────────────────
  const queryPrompt = `You are a ${dbType} expert. Given the schema and question, generate ONLY the query. No explanation, no answer — just the query.

DATABASE SCHEMA:
${schemaStr}

USER QUESTION: "${question}"

Rules:
- ${isMongo ? 'Return a MongoDB JS expression like: db.collection("products").find({}).sort({price:-1}).limit(5).toArray()' : 'Return a SQL SELECT statement only'}
- Single line, no comments
- Only read operations

Respond with ONLY valid JSON:
{"query": "the complete query here"}`;

  let generatedQuery;
  try {
    const raw1 = await callAI([{ role: 'user', content: queryPrompt }]);
    const clean1 = raw1.replace(/```json|```/g, '').trim();
    const parsed1 = JSON.parse(clean1);
    if (!parsed1.query || typeof parsed1.query !== 'string') throw new Error('No query returned');
    generatedQuery = parsed1.query;
    console.log('[Pass 1] Query:', generatedQuery);
  } catch (err) {
    console.error('[Pass 1 failed]', err.message);
    return res.status(500).json({ error: 'Failed to generate query: ' + err.message });
  }

  // ── Execute the query against real DB ────────────────────────
  let results;
  try {
    results = await executeQuery(generatedQuery, dbType, connectionString, maxResults || 100);
    console.log('[Executed] Rows:', results.length);
  } catch (err) {
    console.error('[Execute failed]', err.message);
    return res.status(500).json({ error: 'Query execution failed: ' + err.message });
  }

  // ── PASS 2: AI writes answer from REAL results ───────────────
  const answerPrompt = `You are QueryMind, a friendly data analyst. The user asked: "${question}"

The query was executed and returned these REAL results:
${JSON.stringify(results.slice(0, 20), null, 2)}
Total rows returned: ${results.length}

Write a response using ONLY these actual values — never guess or make up numbers.

Respond with ONLY valid JSON:
{
  "answer": "Friendly 1-3 sentence response using exact values from the results above. Lead with the key finding. Example: 'The most expensive product is MacBook Pro at Rs. 2,49,999, which is 2x the price of the next item on the list.'",
  "visualization": {
    "suggested": true or false,
    "chartType": "bar" or "line" or "pie" or "doughnut" or "scatter" or "none",
    "chartTitle": "Short descriptive title",
    "chartLabelField": "field name to use as X-axis labels",
    "chartValueField": "field name to use as numeric values",
    "reason": "why this chart helps"
  }
}

Visualization rules:
- top N / ranking / comparison → "bar"
- trend over time → "line"  
- proportions / category share → "pie" or "doughnut"
- single value result or text-only → "none"
- if results is empty → answer that no data was found, visualization.suggested = false`;

  try {
    const raw2 = await callAI([{ role: 'user', content: answerPrompt }]);
    const clean2 = raw2.replace(/```json|```/g, '').trim();
    const parsed2 = JSON.parse(clean2);
    res.json({
      isChat: false,
      query: generatedQuery,
      answer: parsed2.answer || 'Query executed successfully.',
      visualization: parsed2.visualization || { suggested: false, chartType: 'none' },
      results
    });
  } catch (err) {
    // Even if pass 2 fails, return results with a basic answer
    console.error('[Pass 2 failed]', err.message);
    res.json({
      isChat: false,
      query: generatedQuery,
      answer: 'Query returned ' + results.length + ' result' + (results.length !== 1 ? 's' : '') + '.',
      visualization: { suggested: false, chartType: 'none' },
      results
    });
  }
});

// Chat endpoint — casual conversation routed through backend (no CORS issues)
app.post('/api/chat', async (req, res) => {
  const { message, history, schema, dbType } = req.body;
  if (!message) return res.status(400).json({ error: 'message required' });

  const schemaCtx = schema && schema.length
    ? 'You are connected to a ' + (dbType || 'database') + ' database with tables/collections: ' + schema.map(t => t.name || t.table).join(', ') + '. '
    : '';

  const systemMsg = schemaCtx + `${process.env.SYSTEM_INSTRUCTION}`;

  // Build message history
  const msgs = [...(history || []).slice(-8), { role: 'user', content: message }];

  try {
    const reply = await callAI(msgs, systemMsg);
    res.json({ reply: reply.trim() });
  } catch (err) {
    console.error('[chat]', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Start ─────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log('QueryMind running at http://localhost:' + PORT + '/agent');
  console.log('AI Provider: ' + (process.env.AI_PROVIDER || 'gemini'));
});
