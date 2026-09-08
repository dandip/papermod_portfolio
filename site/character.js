/** Sprite controller. Atlas: 8 columns, 6 rows, 128px cells.
 * Rows 0–3: idle, walk, wave, chalk. Row 4: watering. Row 5: plant prop.
 * play('wave') plays once and returns to idle. State requests during a wave
 * lower the hand before switching. onstatechange receives {state, phase}.
 */
class PixelCharacter {
  constructor(canvas, src) {
    this.canvas=canvas; this.ctx=canvas.getContext('2d');
    this.state='idle'; this.phase='bounce'; this.x=canvas.width/2;
    this.y=canvas.height-34; this.direction=1; this.scale=2; this.speed=64;
    this.loopScenes=false;
    this.elapsed=0; this.paused=false; this.moving=0; this.last=null;
    this.pending=null; this.lowering=false; this.lowerFrames=[]; this.onstatechange=null;
    this.image=new Image();
    this.ready=new Promise((resolve,reject)=>{this.image.onload=resolve;this.image.onerror=reject;});
    this.image.src=src; this.loop=this.loop.bind(this);
    this.raf=requestAnimationFrame(this.loop);
  }
  static idle=[{f:0,d:.28},{f:1,d:.20},{f:0,d:.36}];
  static walkFrameCount=4;
  static walkFPS=6;
  static wave=[
    {f:0,d:.08,p:'raise'},{f:1,d:.09,p:'raise'},{f:2,d:.09,p:'raise'},{f:3,d:.09,p:'raise'},
    {f:4,d:.10,p:'wave'},{f:5,d:.13,p:'wave'},{f:4,d:.10,p:'wave'},{f:3,d:.12,p:'wave'},
    {f:4,d:.10,p:'wave'},{f:5,d:.13,p:'wave'},{f:4,d:.10,p:'wave'},{f:3,d:.12,p:'wave'},
    {f:2,d:.09,p:'lower'},{f:6,d:.10,p:'lower'},{f:7,d:.10,p:'lower'},{f:0,d:.12,p:'lower'}
  ];
  static sample(sequence,time,loop=false) {
    const total=sequence.reduce((sum,v)=>sum+v.d,0);
    if(loop)time%=total;
    for(const entry of sequence){if(time<entry.d)return entry;time-=entry.d;}
    return null;
  }
  notify(){if(this.onstatechange)this.onstatechange({state:this.state,phase:this.phase});}
  setState(state){
    if(state==='blackboard'||state==='watering'){
      const idleTime=this.state==='idle'?this.elapsed:0,cycle=idleTime%.84;
      const finishBounce=cycle>=.28&&cycle<.48;
      this.boardScene={idleTime,finishBounce,settle:(finishBounce?.48-cycle:0)+.16};
      this.direction=this.x>this.canvas.width/2?-1:1;
    }
    this.state=state;this.phase=(state==='blackboard'||state==='watering')?'settle':state==='wave'?'raise':state==='idle'?'bounce':'step';
    this.elapsed=0;this.lowering=false;this.pending=null;this.notify();
  }
  play(state) {
    if(!['idle','walk','wave','blackboard','watering'].includes(state))throw new Error('Unknown animation: '+state);
    this.paused=false;
    if(this.state==='blackboard'||this.state==='watering'){
      if(state!==this.state)this.pending={state,moving:0};
      return;
    }
    if(this.state==='wave'){
      if(state==='wave')return;
      this.pending={state,moving:0};
      if(!this.lowering){
        const f=this.getPose().frame;
        const routes={0:[0],1:[1,7,0],2:[2,6,7,0],3:[3,2,6,7,0],4:[4,3,2,6,7,0],5:[5,4,3,2,6,7,0],6:[6,7,0],7:[7,0]};
        const steps=routes[f];
        this.lowerFrames=steps.map(f=>({f,d:.09,p:'lower'}));
        this.lowering=true;this.elapsed=0;this.phase='lower';this.notify();
      }
      return;
    }
    this.moving=0;
    if(this.state!==state)this.setState(state);
  }
  move(direction) {
    direction=Math.sign(direction);
    this.play(direction?'walk':'idle');
    if(this.state==='wave'||this.state==='blackboard'||this.state==='watering'){this.pending={state:direction?'walk':'idle',moving:direction};return;}
    this.moving=direction;if(direction)this.direction=direction;
  }
  pause(value=true){this.paused=value;}
  destroy(){cancelAnimationFrame(this.raf);}
  advance(dt){
    if(this.paused)return;
    this.elapsed+=dt;
    if(this.state==='blackboard'||this.state==='watering'){
      // Keep the active gesture on a whole number of two-pose cycles.
      // Props stay visible; the entrance and exit are not replayed.
      if(this.loopScenes&&!this.pending&&this.elapsed>=this.boardScene.settle+2.72){
        this.elapsed=this.boardScene.settle+.72+(this.elapsed-this.boardScene.settle-.72)%2;
      }
      const t=this.elapsed-this.boardScene.settle;
      const phase=t<0?'settle':t<.45?'appear':t<.72?'raise':t<3.12?(this.state==='watering'?'water':'write'):t<3.57?'lower':t<4.02?'disappear':'rest';
      if(t>=4.2){const next=this.pending||{state:'idle',moving:0};this.setState(next.state);this.moving=next.moving;if(this.moving)this.direction=this.moving;}
      else if(this.phase!==phase){this.phase=phase;this.notify();}
    }
    if(this.state==='wave'){
      const entry=PixelCharacter.sample(this.lowering?this.lowerFrames:PixelCharacter.wave,this.elapsed);
      if(!entry){
        const next=this.pending||{state:'idle',moving:0};
        this.setState(next.state);this.moving=next.moving;
        if(this.moving)this.direction=this.moving;
      }else if(this.phase!==entry.p){this.phase=entry.p;this.notify();}
    }
    this.x=Math.max(60,Math.min(this.canvas.width-60,this.x+this.moving*this.speed*dt));
  }
  getPose(){
    if(this.state==='blackboard'||this.state==='watering'){
      const t=this.elapsed-this.boardScene.settle,sceneRow=this.state==='watering'?4:3;
      if(t<0)return {row:0,frame:this.boardScene.finishBounce?PixelCharacter.sample(PixelCharacter.idle,this.boardScene.idleTime+this.elapsed,true).f:0};
      if(t<.45||t>=3.57)return {row:0,frame:0};
      if(t<.72)return {row:sceneRow,frame:0};
      if(t<3.12)return {row:sceneRow,frame:1+Math.floor((t-.72)*4)%2};
      if(t<3.27)return {row:sceneRow,frame:2};
      if(t<3.42)return {row:sceneRow,frame:0};
      return {row:sceneRow,frame:3};
    }
    if(this.state==='idle')return {row:0,frame:PixelCharacter.sample(PixelCharacter.idle,this.elapsed,true).f};
    if(this.state==='walk'){
      const frame=Math.floor(this.elapsed*PixelCharacter.walkFPS)%PixelCharacter.walkFrameCount;
      return {row:1+Math.floor(frame/8),frame:frame%8};
    }
    const entry=PixelCharacter.sample(this.lowering?this.lowerFrames:PixelCharacter.wave,this.elapsed);
    return {row:2,frame:entry?entry.f:0};
  }
  loop(time){
    const dt=this.last===null?0:Math.min((time-this.last)/1000,.05);this.last=time;
    this.advance(dt);this.draw();this.raf=requestAnimationFrame(this.loop);
  }
  draw(){
    const ctx=this.ctx;
    ctx.clearRect(0,0,this.canvas.width,this.canvas.height);
    if(!this.image.complete||!this.image.naturalWidth)return;
    const {row,frame}=this.getPose();
    ctx.imageSmoothingEnabled=false;
    if(this.state==='blackboard')this.drawBoard();
    if(this.state==='watering')this.drawPlant();
    ctx.save();ctx.translate(Math.round(this.x),Math.round(this.y));
    ctx.scale(this.direction*this.scale,this.scale);
    if(this.state==='idle'||((this.state==='blackboard'||this.state==='watering')&&row===0)){
      // Shoes are a fixed sprite layer; only the bent-knee/body frames change.
      // No translation, scaling, rotation, or artificial bob on the idle pose.
      const shoeLine=109;
      ctx.drawImage(this.image,frame*128,0,128,shoeLine,-64,-120,128,shoeLine);
      ctx.drawImage(this.image,0,shoeLine,128,128-shoeLine,-64,shoeLine-120,128,128-shoeLine);
     }else if(this.state==='watering'&&row===4){
      // Reuse the approved neutral idle head, not the regenerated watering face.
      ctx.drawImage(this.image,0,0,128,64,-64,-120,128,64);
      ctx.drawImage(this.image,frame*128,576,128,40,-64,-56,128,40);
      ctx.drawImage(this.image,0,616,128,24,-64,-16,128,24);
    }else if(this.state==='blackboard'&&row===3){
      // Keep the teaching stance completely fixed. Only swap the chalk-arm region.
      const armPath=()=>{
        ctx.moveTo(82-64,36-120);ctx.lineTo(128-64,36-120);
        ctx.lineTo(128-64,101-120);ctx.lineTo(75-64,101-120);
        ctx.lineTo(75-64,61-120);ctx.lineTo(82-64,53-120);ctx.closePath();
      };
      ctx.save();ctx.beginPath();ctx.rect(-64,-120,128,128);armPath();ctx.clip('evenodd');
      ctx.drawImage(this.image,0,384,128,128,-64,-120,128,128);ctx.restore();
      ctx.save();ctx.beginPath();armPath();ctx.clip();
      ctx.drawImage(this.image,frame*128,384,128,128,-64,-120,128,128);ctx.restore();
    }else {
      // Use each aligned pose's own face and hair for subtle expression changes.
      ctx.drawImage(this.image,frame*128,row*128,128,128,-64,-120,128,128);
    }
    ctx.restore();
  }
  drawPlant(){
    const t=this.elapsed-this.boardScene.settle;
    if(t<0||t>=4.02)return;
    const ease=v=>{v=Math.max(0,Math.min(1,v));return v*v*(3-2*v);};
    const visible=t<.45?ease(t/.45):t<3.57?1:1-ease((t-3.57)/.45);
    const ctx=this.ctx,d=this.direction;
    ctx.save();ctx.globalAlpha=visible;ctx.translate(this.x,this.y);ctx.scale(d,1);
    ctx.drawImage(this.image,0,640,128,128,54,-132,144,144);
    if(t>=.80&&t<3.12){
      // Anchor the landing point to exposed medium inside the pot's top ellipse.
      const soilX=54+52*(144/128),soilY=-132+79*(144/128);
      for(let i=0;i<7;i++){
        const u=((t-.8)*2.1+i/7)%1;
        const x=84+(soilX-84)*u,y=-48+(soilY+48)*(u*.35+u*u*.65);
        ctx.fillStyle=i%2?'#a6e3ed':'#d6f7ee';ctx.fillRect(Math.round(x),Math.round(y),2,3);
      }
      const splash=Math.floor(t*8)%2;
      ctx.fillStyle='#a6e3ed';ctx.fillRect(Math.round(soilX-2-splash),Math.round(soilY-1),2,2);ctx.fillRect(Math.round(soilX+2+splash),Math.round(soilY-2),2,2);
    }
    ctx.restore();
  }
  drawBoard(){
    const t=this.elapsed-this.boardScene.settle;
    if(t<0||t>=4.02)return;
    const smooth=v=>{v=Math.max(0,Math.min(1,v));return v*v*(3-2*v);};
    const visible=t<.45?smooth(t/.45):t<3.57?1:1-smooth((t-3.57)/.45);
    if(!this.boardTexture){
      const b=document.createElement('canvas');b.width=124;b.height=88;const c=b.getContext('2d');
      const rect=(x,y,w,h,color)=>{c.fillStyle=color;c.fillRect(x,y,w,h);};
      // Small stepped timber frame, inset slate, and a projecting chalk tray.
      rect(1,1,122,85,'#30271f');rect(2,0,120,85,'#513b2a');
      rect(3,1,118,3,'#b18a59');rect(2,3,3,78,'#947044');
      rect(119,3,3,79,'#674a31');rect(5,4,114,77,'#122e29');
      rect(6,5,112,75,'#1c4238');rect(7,6,110,1,'#2a5042');
      rect(7,7,1,72,'#24493e');rect(116,7,1,72,'#173a32');
      // Deterministic faint chalk wear, kept behind the writing.
      for(let i=0;i<45;i++){
        const x=9+(i*37)%105,y=9+(i*19)%68;
        rect(x,y,1+(i%3),1,i%2?'#24483d':'#20453b');
      }
      rect(8,76,24,1,'#294c40');rect(83,74,23,1,'#25483d');
      // A bitmap alphabet keeps the chalk crisp at the character's pixel scale.
      const glyphs={
        A:['010','101','111','101','101'],C:['011','100','100','100','011'],
        D:['110','101','101','101','110'],E:['111','100','110','100','111'],
        G:['011','100','101','101','011'],I:['111','010','010','010','111'],
        L:['100','100','100','100','111'],M:['101','111','111','101','101'],
        N:['101','111','111','111','101'],Q:['010','101','101','111','011'],
        R:['110','101','110','101','101'],S:['011','100','010','001','110'],
        T:['111','010','010','010','010'],U:['101','101','101','101','111'],
        V:['101','101','101','101','010'],X:['101','101','010','101','101'],
        'δ':['010','001','011','101','011'],'γ':['000','101','010','010','010'],
        'α':['000','011','101','101','011'],
        '(':['010','100','100','100','010'],')':['010','001','001','001','010'],
        '=':['000','111','000','111','000'],'+':['000','010','111','010','000'],
        '-':['000','000','111','000','000'],'←':['00100','01000','11111','01000','00100'],
        ',':['000','000','000','010','100'],"'":['010','010','000','000','000']
      };
      const chalk=(text,x,y,color='#e4e6d1')=>{
        for(const ch of text){const g=glyphs[ch]||glyphs[ch.toUpperCase()];
          if(g)g.forEach((line,dy)=>[...line].forEach((pixel,dx)=>{if(pixel==='1')rect(x+dx,y+dy,1,1,color);}));
          x+=ch==='←'?6:4;
        }
      };
      chalk("δ = r + γ max Q(s',a')",11,31);
      chalk('- Q(s,a)',27,40);
      // Box the update rule and add a tiny rising value trace beside it.
      rect(10,54,71,1,'#7d9b84');rect(9,55,1,15,'#819e85');
      rect(11,70,69,1,'#a3b49a');rect(81,55,1,14,'#8aa68c');
      chalk('Q ← Q + αδ',17,60,'#f0edcf');
      chalk('V',91,48,'#aabda1');rect(91,58,1,13,'#78947e');rect(91,71,22,1,'#78947e');
      [[94,68,3,1],[96,65,1,4],[97,65,4,1],[100,62,1,4],[101,62,4,1],[104,59,1,4],[105,59,7,1]].forEach(([x,y,w,h])=>rect(x,y,w,h,'#c2cba7'));
      rect(3,81,118,2,'#bb905b');rect(1,83,122,3,'#805d39');rect(2,86,121,2,'#392b21');
      rect(7,84,26,1,'#9b7145');rect(46,84,20,1,'#6e4f32');
      // Felt eraser and two pieces of chalk on the ledge.
      rect(88,78,13,4,'#303734');rect(89,77,11,2,'#96784e');rect(89,80,11,2,'#a5aaa0');
      rect(105,80,7,2,'#efecdb');rect(114,79,3,2,'#d6d9c3');
      [[3,2],[119,2],[3,79],[119,79]].forEach(([x,y])=>{rect(x,y,1,1,'#dec391');});
      this.boardTexture=b;
    }
    const ctx=this.ctx,w=248,h=176;
    const x=this.direction===1?this.x+54:this.x-54-w;
    ctx.save();ctx.globalAlpha=visible;
    ctx.drawImage(this.boardTexture,Math.round(x),Math.round(this.y-215+(1-visible)*18),w,h);
    ctx.restore();
  }
}
window.PixelCharacter=PixelCharacter;
