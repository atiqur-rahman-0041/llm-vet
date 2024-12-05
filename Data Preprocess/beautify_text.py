
patch_text = """"hunk2: export class MongoDBStore extends BaseStore<string,Uint8Array>{.toArray();const encoder=new TextEncoder();\nreturn retrievedValues.map((value)=>{if(!(\"value\"in value)){\nreturn undefined;}\nif(value===undefined||value===null){\nreturn undefined;}else if(typeof value.value===\"object\"){return encoder.encode(JSON.stringify(value.value));"""
patch_text = patch_text.replace("\\n", "\n")
print(patch_text)