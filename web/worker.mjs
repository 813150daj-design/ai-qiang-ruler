import {analyze} from './engine.mjs';
let loaded;
async function load(){
  const get=async path=>{const r=await fetch(new URL(path,import.meta.url));if(!r.ok)throw Error(`资源加载失败 (${r.status})`);return r.json();};
  loaded ||= Promise.all([get('./model.json'),get('./checks.json')]);
  const [model,checks]=await loaded;
  if(model.format!=='aiqiang-model'||model.version!==1)throw Error('模型格式不支持');
  return [model,checks];
}
self.onmessage=async({data})=>{try{const [model,checks]=await load();self.postMessage({type:'ready'});self.postMessage({id:data.id,result:analyze(data.text,model,checks)});}catch(error){loaded=null;self.postMessage({id:data.id,error:error.message});}};
