# Values-SFT candidate review

521 candidates. Built by `scripts/build_values_prompt_candidates.py`; provenance in `DECISIONS.md` (same directory).

## What this is

Candidate **user turns** for a fine-tuning corpus. Each one is a real person describing an ordinary situation and asking what to do. They came from r/AmItheAsshole (via AI2's Scruples release) and have been filtered by script; you are the human pass.

Later, each of these gets four different assistant replies written for it — the same situation answered from four different value priorities. That is the experiment. **You do not need to judge that part**, and you do not need any background on the study.

## What to decide

Each candidate has a `**Verdict:**` line, pre-filled with `KEEP`. Change it only when something is wrong. Leave the prompt text alone unless you are fixing it (see below).

| Verdict | Use when |
|---|---|
| `KEEP` | Fine. An ordinary, low-stakes situation, and it reads like something a person actually sent. |
| `DROP: unsafe` | **Not low-stakes.** Anything where a confident recommendation could hurt someone — self-harm, abuse, a medical or mental-health decision, legal jeopardy, money trouble that is real hardship, or a situation turning on a child at risk. When in doubt, drop it. |
| `DROP: unnatural` | Doesn't work as a standalone message — incoherent, refers to context that isn't there, is a rant with no actual decision in it, or is so garbled it can't be answered. |
| `DROP: no fork` | There is only one sensible answer. A good candidate is one where reasonable people could genuinely advise different things. |
| `UNSURE` | Anything you'd want a second opinion on. These come back to Hilary — using it liberally is fine and costs nothing. |

**The `unsafe` pass is the important one.** Automated filters already ran and they leaked repeatedly — a roommate self-harming, a pet being put down, a fifteen-year-old — each caught only by someone reading it. That is the job here.

## Fixing rather than dropping

If a candidate is nearly fine, edit the text inside its ```` ```text ```` fence and leave the verdict as `KEEP`. Keep the fence markers. Every edit is automatically re-checked afterwards, so you cannot break anything silently.

## Notes

- The **Title** line is the original post title, shown to help you skim. It is not part of the prompt.
- Most prompts end with "Would I be wrong if...?" — that is generated from the title and is meant to be there.
- Reddit register (`[18F]`, "tl;dr", the confessional tone) is expected and is **not** a reason to drop.
- Target is roughly 250 keepers out of 521. Being strict is the right instinct.


---

## 001 · `007ffd0da91638b6` · 1921c · ask=title

> **Title** (not part of the prompt): WIBTA if I clean my brother's room without his permission when he is out.

```text
Both still living with parent because yay, no job! Brother gets SSI cause handicapped from waist down, so wheelchair.

He never cleans his room, he usually has a huge pile of clothes next to an overfilled diaper pail. (He usually wants me to take that out, but I make him do that), that gets taken care of super late, so its usually a stench of dirty laundry and used diapers. He usually shuts the door when he heads out, but when he comes back the door comes open and there's the stench again! I've asked him to take his diaper pail out earlier, but he always waits til garbage day or when it gets really overfilled. I've asked him to take the laundry to the hallway (we put laundry baskets there) to reduce the stink in his room, but he always tells me "It will still stink!" It will, but it won't be overpowering.

He has piles of stuff everywhere in his room, most of it is Magic the Gathering cards, plastic bags and recyclables.

I told him of an idea, "Why don't we sort out the Magic the Gathering cards, anything we have more than a playset of (4 of the same card) we take the rest and you sell them to a store (they give $5 per 1000)" at some point I might just start sorting when he's out. He constantly gets cards from his friends that they don't need, so it's only gonna add up.

Plastic bags and recyclables I can remove quickly.

He basically has a bunch of mess and he has empty floor leading straight from his computer to his bed.

If I suggest it when he's home, he gets mad and says he'll do it He has never done anything like that for the past year. The only time he has cleaned his room was when he had an aide and was basically forced to clean his room.

If he won't do it, maybe I will, and if I start, maybe he might too. Usually he complains about the process of doing things, "It would be hard for me to do."

Would I be wrong if I clean my brother's room without his permission when he is out.?
```

**Verdict:** KEEP

---

## 002 · `00a769573a7bd951` · 1360c · ask=body+title

> **Title** (not part of the prompt): WIBTA For calling animal control on my relatives?

```text
Apologizing in advance for any format issues, I'm on mobile.

Alrighty, so here's the info. I really love these relatives of mine, they're really good people, a nice couple, but there's just one problem. They're borderline animal hoarders. (I will be referring the Husband as X and the Wife as Y)

It almost feels like every year that passes, they get a new dog. As of now they have 8 dogs (previously 9) and they had to get rid of one to Y's father. I feel like this animal hoarding is getting out of control and is effecting the household.

The house constantly reeks of urine and the dogs like to drink the green pool water in the backyard. Their house has scattered clothes around the livingroom and hallways and I feel like if maybe if they had less dogs that they would actually put effort into cleaning their house. I've started to hate visiting because it just smells so bad and there's dog hair everywhere.

I'll be honest, I dont WANT to call anyone to take the dogs away, the dogs are cute, but I cant confront them about this issue without X and Y taking it personally. They have a tendency to have a "us against them" mentality with family when any of us try to offer advice.

Should I just let them be and wait for the dogs to just kick the bucket or should I do something about it???

Would I be wrong for calling animal control on my relatives?
```

**Verdict:** KEEP

---

## 003 · `00b74e2b7bf80d62` · 1594c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I rescinded financial help to my Dad?

```text
My dad is an honest, hardworking guy, but has always been financially unstable, running his own business with employees. I'm 27 (only child) and he has never asked me for help until now. Last week he called me very upset and told me that he only had $300 in his account and needed help to pay bills. (He also has no CC because of his credit, so this is his only money). He asked for $2000–$3000. With that little in his account obviously I wanted him to pay his bills and also be able to eat, get gas, etc. I sent him a check for $2500 and figured after this hurdle was out of the way, we could figure out a long term plan. Tonight we spoke again and he told me he went to his CPA and realized he actually has $4500. He seemed to think now everything is ok, but still needed my check to pay his employees and get more established. I asked him, if he has access to his bank account, how did he not realize this was the case? He said that his administrative employee told him he only had $300 and wouldn't be able to pay his employees, just to be mean to him. You'll have to take my word that he's an honest guy, just clueless and naive. I don't think he was trying to trick me, but I feel that the situation is unfair to me because I gave him that money under the belief that he wouldn't be able to pay his mortgage or eat without it. With the money that he has, I could provide a smaller amount or help him figure out a better long-term plan. The check is still in the mail to him. I would obviously give him a heads up about it first.

Would I be wrong if I rescinded financial help to my Dad?
```

**Verdict:** KEEP

---

## 004 · `018337648f36e1a6` · 1092c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for Schrodinger's Gift?

```text
A friend and I are planning on going to a concert together. In the past, we have paid for our own tickets. For this concert we discussed getting VIP tickets, but put off the decision until we knew the cost/benefits of a VIP ticket vs a normal one. When I got more information, I passed along that info but haven't heard back (we are both at work). I didn't want to miss out on the VIP tickets, so I bought them. I don't want him to feel obligated to pay for the upgrade if he didn't budget for it. At the same time, if he's cool with paying me back it would be nice to have the cash. So I haven't told him that I bought them, and am waiting to hear back to determine how to handle the split. If he greenlights the VIP ticket, we can split the cost 50/50. If he doesn't want to pay extra, I bought them being fully okay covering the upgrade. If I told him before getting a response, he is the type of person that would definitely refuse to let me offer it as a gift. So it feels weird having it simultaneously be a possible gift and possible not gift.

Would I be wrong for Schrodinger's Gift?
```

**Verdict:** KEEP

---

## 005 · `01e3260b3da33299` · 1296c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I ratted out my close friend for not doing any work on a project?

```text
FTP, LTL, on mobile, sorry in advance. I am in a group project with 4 others, two of which are in my immediate circle of friends, let’s call them T and G. One group member, who is new to my school and not very well liked (we’ll call her B), has not done any work on the project. My other group mates and I gave her plenty of notice about what her role was and when it had to be completed by, but I ended up having to finish it for her (I have done the majority of the work on this project so far). T reported it to the teacher, and B will most likely get in trouble as she is about to do it again for the second portion of this project. I know this does not make me an asshole, as she really has done no work. My other close friend, G, has also not done any work on the project. She was assigned a portion of the first half of the project, and I ended up completing it as well. She took on no role in the second half of the project. It was not reported because she is best friends with T, and T did not want to get G in trouble. Like I said, I have done most of the work on this project while G and B have not done anything and T has made some contributions with my help (my other group mate is great though).

Would I be wrong if I ratted out my close friend for not doing any work on a project?
```

**Verdict:** KEEP

---

## 006 · `022eb449e079111a` · 857c · ask=title

> **Title** (not part of the prompt): WIBTA if I asked for a higher pay even though I already agreed to a lower one?

```text
So about a week ago one of my family friends asked me to take care of her dog and 7 parrots (yes, really) starting this evening and ending Sunday morning. Since I'm an absolute dumbass, I agreed to a pay of $30. For the entire weekend.

Basically I'd be staying the night and feeding her parrots (which are all assholes) in the morning, afternoon, and evening, so a total of 5 times. I'd also walk her dog, but she's an absolute sweetheart and I have no problem with that.

However, once I told my mom about this she was super surpised and told me that pay was abysmal. I kind of agree with her now and I want to ask for a higher pay of maybe $50-$60 for the weekend.

Is that a shitty thing to do or am I justified in asking for a higher pay? Thanks for the advice.

Would I be wrong if I asked for a higher pay even though I already agreed to a lower one?
```

**Verdict:** KEEP

---

## 007 · `04180f20c64d6512` · 592c · ask=title

> **Title** (not part of the prompt): WIBTA if I asked my dad to not invite his girlfriend on a trip we've been planning?

```text
My dad's girlfriend is a great person to be around and I like her a lot, but my dad lives in Colorado and I, Iowa, so I don't get to see him often. We are planning a week-long fishing trip in Colorado this summer. Camping, fishing, drinking, what you would expect of a guys weekend. I just worry that my dad will invite his girlfriend on this trip. Like I said, I like her, I just want this time with my dad. Time to bond as I haven't lived in the same town as him in the last decade. Thanks all.

Would I be wrong if I asked my dad to not invite his girlfriend on a trip we've been planning?
```

**Verdict:** KEEP

---

## 008 · `0481fea5501918d7` · 463c · ask=title

> **Title** (not part of the prompt): WIBTA if I wore African clothing

```text
There is a really cool company called "no tribe clothing." It's African American owned and operated. I also love their clothes. It's African textile clothing, kente cloth like style, very different from European style clothing.

But, I'm afraid of being called out for cultural appropriation as a mostly white person. I don't want to offend. I just like supporting Black companies, especially if I like their product.

Would I be wrong if I wore African clothing?
```

**Verdict:** KEEP

---

## 009 · `04ffd3407aa45da5` · 663c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I were to lock out my roommates of the wifi to keep their noise down?

```text
Hi, I'm currently in a situation with 2 of my 5 roommates where they are up playing Fortnite (which in and of itself isn't an issue) until 2-3 AM. The problem comes in is that they yell constantly during that time. I don't have an issue with it during a certain time of the day but at this point it is getting kind of ridiculous. So, as the sole controller of the router and internet service to the house, I was debating with my other 2 roommates on whether I should turn off wifi access after certain time so that they can't play Fortnite until late and keep everyone up.

Would I be wrong if I were to lock out my roommates of the wifi to keep their noise down?
```

**Verdict:** KEEP

---

## 010 · `05226cce0184caee` · 893c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I went to my boss because I suspect my co-worker isn’t showing up for work?

```text
There are only a few people who work in my position. It’s a pretty “low on the totem pole” job, but it requires us to be at work early in the morning - there is a part of our job that literally cannot be sufficiently completed unless you’re there at a certain time. My co-workers and I rotate shifts between a few locations, one of which my boss rarely visits. I have good reason to believe that my co-worker doesn’t show up, or shows up very late, when he’s in that location. It bothers me for a lot of reasons, mostly because it demonstrates a blatant lack of work ethic and it undermines a key purpose, value, and quality of our job and the work we do, and has the potential to create more work for me (albeit marginally). I don’t want to talk to my other co-workers because it’s not my place.

Would I be wrong if I went to my boss because I suspect my co-worker isn’t showing up for work?
```

**Verdict:** KEEP

---

## 011 · `05cc626f6bd8a0fa` · 1224c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA to demand my roommates sister pays rent?

```text
My roommate and I have been bestfriends since kindergarten and hes a fantastic roommate. However about 6 months ago him, a third person, and I were going to get an apartment together but after some unfortunate events not relevant to the story my roommate and I got stuck on the lease while the third person bailed. So now the 2 of us are paying rent for a 3 person apartment and I cant handle that financially for much longer and we've been trying to find a 3rd person to take some of the pressure off. Flash forward to about 2 months ago, my roommates sister fell on some hard times and moved in with us (not on the lease) and she doesnt seem to have any intentions of leaving anytime soon. She pays about $100 in rent when if split equally between the 3 of us it should be around $470. She has a part time job at a local grocery store (though she has told me she could get a full time position there if she wanted.) And I know it was inappropriate of me, but the other day she opened her paycheck in front of me and I glanced at it and it was around $600 (so $1200 a month). Because again I cant pay for essentially a 3 bedroom place split 2 ways for much longer.

Would I be wrong to demand my roommates sister pays rent?
```

**Verdict:** KEEP

---

## 012 · `07379001a1f19481` · 842c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for making someone come to me to get their watch back?

```text
I went on a couple dates with a guy almost a year ago. After the second date, he accidentally left his apple watch behind at my house. I texted him to hang out a couple more times, but he kept avoiding me, and eventually he just ghosted. I texted him several times over the course of a few months asking if he wanted the watch back, and he just never responded. Last week he texts me out of the blue asking if I have his watch, and I say I do. Then he asks if we can meet for coffee in the city or if I can mail it to him. Frankly, I'm irritated that I should have to travel 45+ minutes or go on a whole errand to the post office after this guy ghosted me; he clearly didn't even want the apple watch that much, since he didn't bother getting it back for almost a year.

Would I be wrong for making someone come to me to get their watch back?
```

**Verdict:** KEEP

---

## 013 · `0739bedf630f2d68` · 564c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for helping a girl with school work for a blowjob

```text
Before anyone gets the wrong impression I’m the same age as her (both 15) Basically I was casually talking to this girl who I’m mates with and she’s been on holiday last 2 weeks and I told her we’ve got mocks for GCSEs coming up. She said she was fucked and I agreed with her. Then she asked me I could help her jokingly then said wat do I get for it. Then she said I’ll give u a blowjob for it. At this point I was shocked and asked if she’s being serious and she said yeah so I said of course.

Would I be wrong for helping a girl with school work for a blowjob?
```

**Verdict:** KEEP

---

## 014 · `073d0e85295cf251` · 963c · ask=body+title

> **Title** (not part of the prompt): WIBTA for reporting my school shop teacher to the principal because of an unsafe shop?

```text
He doesn't know how to use the machinery in the shop so just leaves it to students. He doesn't have eye protection available so I had to use safety glasses I had with me. He leaves students alone in the shop with machines that are very dangerous and could cause massive injuries if not used safely. He leaves students alone in the shop with machines that cost several tens of thousands of dollars. 
I could've had a serious accident today with nobody to even know I was hurt because he left for the other room without notifying me and I pressed the wrong button on a lathe at 300RPM. 
I am considering reporting him to the principal (which would likely put his job on the line because he's endangering lots of people and risking breaking expensive machinery) because I feel extremely unsafe working in the shop. 
Am I overreacting? Am I the one who's in the wrong?

Would I be wrong for reporting my school shop teacher to the principal because of an unsafe shop?
```

**Verdict:** KEEP

---

## 015 · `084a3e28b56f0d8a` · 1011c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for going to meet with a female friend, after having just organised a date with another friend?

```text
Hi, This is an odd one but I just want to make sure I'm not overstepping by doing this. So recently, I (M) asked a girl I have been talking to out on a date. She was quite excited about it and we have arranged to go this saturday night. I am quite excited about it as well, as she seems like an amazing person. However, last night, I was talking with another female friend who went out with one of my friends, and she has been quite down after some events happening between them. She feels 'used' by him, and after hearing he might be getting back in with his ex, was quite upset by it. I said I'll try to meet up with her (before my date) and just make sure she is doing well, but I'm worried that by doing this, it would be really disrespectful to the girl I asked out. Should I possibly mention it to the girl just so she understands why I'm going to meet her and doesn't take it the wrong way?

Would I be wrong for going to meet with a female friend, after having just organised a date with another friend?
```

**Verdict:** KEEP

---

## 016 · `097d06b2961bf55d` · 1222c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I lied to a friend about not being able to go with him to a place just to go with another friend?

```text
So, one of my friends went on a trip with me to Algarrobo (I live in Chile, and this is like going from Orlando to West Palm Beach) in July for 2 weeks and we had a pretty good time. A few weeks ago we were talking about making another trip to this place for the good old times, and I agreed, but one of my friends (that is actually my best friend and I’ve been knowing him since babies) quit his job, and that means he’s now free to go anywhere and he hasn’t gone on a trip with me since 2015 if I recall correctly, and I would love spending time with him and do a lot of stuff, but I promised my other friend to go with me already, and my deadline is the 6th of January. I’m not the kind of guy that can go and tell people (specially close friends) and tell them that I no longer want to do something with them because I want to do it with someone else. #tl;dr: I promised one of my friends that already went on a trip with me not long ago to go again, but best friend wanted to go for a long time so I want to go with best friend and i want to lie to other friend because i don’t know how to tell him

Would I be wrong if I lied to a friend about not being able to go with him to a place just to go with another friend?
```

**Verdict:** KEEP

---

## 017 · `0a5d17be0ebc23a1` · 520c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I hang out with someone else tonight?

```text
I previously made plans with someone to chill tonight, but now have found out someone I haven't seen for over a month, and likely won't see for several months after tonight, will be passing through town. Now they have asked me to hang out tonight. Person 1 I live like a 35min drive from, so I'd be able to hang out wit them plenty of other times, but they would probably be dissapointed if not upset that I'd be picking someone else to hang out with over them.

Would I be wrong if I hang out with someone else tonight?
```

**Verdict:** KEEP

---

## 018 · `0c117266ce35bdcb` · 408c · ask=title

> **Title** (not part of the prompt): WIBTA if I asked an artist if I can sell their work?

```text
Hear me out here, I bought these nice phone cases from an artist. They are decoden and lovely. However I no longer have that phone type. So I wanted to sell the ones I can't use so they can go to a good home and I can purchase a new one from the artist. Would I be an asshole for evening asking the artist if I can do this?

Thanks in advance.

Would I be wrong if I asked an artist if I can sell their work?
```

**Verdict:** KEEP

---

## 019 · `0ccea4ca114231e8` · 768c · ask=title

> **Title** (not part of the prompt): WIBTA for not wanting to date a guy who refuses to use his turn signal?

```text
I mean everyone has times where they won't bother to signal because maybe they're alone on the road and need to lane change, or they're turning right on red, or maybe it's a protected left, or you're in a parking lot, but this guy has an attitude about his turn signal. "I don't need to use it, I know what I'm doing. If other people know how to drive, they don't need me to use it either." And then he gets road rage when he's trying to move over and the car next to him won't move forward or yield. 
I'm not looking to just date, I'm shopping for a husband, and I don't want a husband whose selfishly going to put people in danger, get alot of tickets, and fuck up our CLUE reports.

Would I be wrong for not wanting to date a guy who refuses to use his turn signal?
```

**Verdict:** KEEP

---

## 020 · `0d03cce7d3fe54e7` · 666c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I told people I was black even though I'm technically not

```text
Okok I know the title sounds bad but here's context 
I'm mixed race. Dads black from Trinidad, moms white from south Africa. Im REALLY light skinned and when I want to a predominantly black school people labeled me "white girl" since there were only 2 white girls at the school and one other light skinned person. I almost never correct people when they say I'm white but that is really my fault. I'm in a new country now where I don't know anyone so I havnt claimed I'm white in this country yet. Even though I'm mixed and light, would it be appropriate for me to call myself black ???

Would I be wrong if I told people I was black even though I'm technically not?
```

**Verdict:** KEEP

---

## 021 · `0dc385ee47603140` · 567c · ask=body

> **Title** (not part of the prompt): WIBTA (f25) for forcing other bridesmaid to buy more expensive dress?

```text
The situation: I went bridesmaid shopping with bride to be/her maid of honor today. Both bridesmaid and the bride fell in love with a 200 dress. The third bridesmaid had tried on this dress previously but didn’t want to spend 200 dollars and requested we keep the dress under 100 dollars. We really live this dress though and other options the third girl picked out have to be ordered online so there is no way of knowing how they fit. Also, the dress we picked is in sale now at 30$ off, making the dress 170. WWBTA for asking her to pay 70 more for the nicer dress?
```

**Verdict:** KEEP

---

## 022 · `0ec5c906f3c72a96` · 411c · ask=title

> **Title** (not part of the prompt): WIBTA if I apply to two internships at the same company?

```text
I worked at this company last year and I believe that they would definitely hire me back for the same position, but I already applied and interviewed for a different internship at the same company. I'm wondering if it's a dick move to apply to the old position and potentially turn it down if I'm offered the other one that I interviewed for.

Would I be wrong if I apply to two internships at the same company?
```

**Verdict:** KEEP

---

## 023 · `0f8d743d17eb15c3` · 1764c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I called in a noise complaint on my landlord for playing music?

```text
So a little context, I am renting a room from a place I found on craigslist. The landlord currently lives there and there is one other room aside from mine that he is trying to rent out. I was aware of this going in, but I voiced my strong preference against rooming with couples, anyone under 25, anyone older than 40, or women (can't stand their hair everywhere!). He seemed to take this into consideration and ran potential tenants by me as they reached out to him. But none of them really fit my preferences and it ends up taking an extra month. I realize he is losing money or whatever but I have to live with the person too! So anyway he starts to get irritated with me, saying that I am being too picky about roommates, and acting entitled about the property, blah blah blah. So he ends up telling me I am no longer a fit for the property and that I need to find a new place immediately. Wtf. I remind him I have a right to 30 days notice, which he obliges to but tbh I am really upset about the whole situation now. And here's where the current situation comes in: He eventually stopped asking me for my opinion and found a guy to move in, but I don't like him at all. He left a dish in the sink and used the bathroom at like 630 in the morning. Then around 10:30 am he is like blasting rap music from his phone speaker while he was on his way out to work or whatever. I was already awake doing laundry but I found it extremely irritating and disrespectful. I don't have the new guys information but I do want to do something about this since I have to live here another 3 weeks so I figure I'll just call in a noise complaint about the property since I'm leaving anyway.

Would I be wrong if I called in a noise complaint on my landlord for playing music?
```

**Verdict:** KEEP

---

## 024 · `0f9adb1abcad54a3` · 818c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I didn’t do my coworkers work for her?

```text
The title is loaded. So my coworker works very long hours (7 am  until around 5 pm) and it seems like she is constantly overburdened with work. I on the other hand work around 7 am to 3 pm and my job is operations, so if things are quiet I won’t be so busy. There is an aspect of my job that requires her to input changes to data so my “thing” can capture the changes. This means I cannot be done for the day until she does this. This is an everyday routine. Lately I’ve been offering to input the data for her because I can see she is swamped and I don’t want to be held here for this one small thing. Here’s the thing though, she’s starting to send over data that is not for tomorrow, but next week. I feel like she’s starting to take advantage of my help.

Would I be wrong if I didn’t do my coworkers work for her?
```

**Verdict:** KEEP

---

## 025 · `0fef488692777c5e` · 2122c · ask=body+title

> **Title** (not part of the prompt): WIBTA IF I WOULDN'T COME TO TAKE MY DRESS AND WOULDN'T PAY the tailor???

```text
I came to a tailor to make my dress for my engagement. She said it would take 2 weeks, I could take it on March 01st, I MADE A DEPOSIT OF 1/3 for it (Im sure it was enough for the clothes and decorations). It was expensive but because I hoped it would be great so I didn't care much about the money. The dress must be ready before March 07 as I need to go home on March 08 and my engagement is on March 09.

Why I chose her? I live an hour driving from my hometown, where the engagement is hosted, and the tailor was around 10 mins from my place. I chose her because the first time I came to her she was very helpful so I trusted her, and because she is near me so I can come and try on early and make any change if needed.

And here the story began... As I was busy on March 01, I came to her on March 02, thought that it was ready. BUT IT DIDN'T. She said that she thought the due date was March 04, the dress needed some decorations, it would take some hours and will be ready on March 03. I was very angry. On March 03, I came to her, the dress was ugly, the decorations were terrible so I asked her to change it. I told her that I would come back on March 05. As on March 05 I was busy again so I texted her saying that I couldn't come, I would come on the next day, March 06, she said it was fine, no problem. BUT yesterday, March 06, she wasn't there, I called her, she said she was away and couldn't come back immediately as she....needed to buy some flowers for the Woman's Day (her daughter owned a flower shop next to the tailor). Again, I was very angry. She asked me to come again today, March 07.
Today, I called her before I came, she also said sorry for the same reason: she needed to buy the flowers. She said she would call me when she was back.

Im writing this at 07:40 PM and I need to go to my home early tomorrow.

This is over my limitation!!!! I run out of patience.

I had a backup dress. It wasn't the best but I can help me now.

So AM I AN ASSHOLE IF I WOULDN'T COME TO TAKE MY DRESS AND WOULDN'T PAY HER????

Would I be wrong if I WOULDN'T COME TO TAKE MY DRESS AND WOULDN'T PAY the tailor???
```

**Verdict:** KEEP

---

## 026 · `102f601a3886a4ae` · 1350c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for boycotting Thanksgiving dinner?

```text
So I am part of a newly blended family. My father married my step mother 3 years ago and as such, we are celebrating Thanksgiving together. I \[23F\] and my sister \[28F\] have inherited two new step siblings \[29M\] \[32F\] whom are both married and one of which who has kids. I am the only one who lives in state. So everyone is traveling to our parents house for Thanksgiving. However, my sister and I have just been told that our aunt (our deceased mother's only sibling) and our grandmother (my father's mother) are not invited to Thanksgiving dinner. My sister and I obviously flipped out a bit since we have spent almost every Thanksgiving and Christmas with them. My dad has been playing mediator with my stepmother and my sister & I. He says that her vision for Thanksgiving is one with only the "core family". This comes from her tumultuous relationships with her two kids, that has finally now calmed down a bit. We've been told that our aunt and grandma can come to the house after my stepmother's kids leave on Friday and that we would have another dinner. That doesn't seem good enough for me. So my sister and I have come to the agreement that if they are excluded that we would not attend Thanksgiving with them and instead have Thanksgiving with our aunt and grandmother instead.

Would I be wrong for boycotting Thanksgiving dinner?
```

**Verdict:** KEEP

---

## 027 · `128241c9e45bc147` · 482c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if i kick the tail light out of drivers who run me off the road when I'm on my bike.

```text
Okay so here's the situation I bike everywhere in my city in all seasons. I obey the rule that cyclists ride in the street with cars, however most drives seem to think that because I'm a cyclist I don't have the same rights of the road as them and try and run me off the road, so I've thought about kicking their tail lights out when at the next traffic light when they do that.

Would I be wrong if i kick the tail light out of drivers who run me off the road when I'm on my bike.?
```

**Verdict:** KEEP

---

## 028 · `12ddf275f18ae74e` · 428c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I don't tip at a restaurant because of my debt?

```text
From all the stories I have read, it seems as if the silent rule is to always tip. But what if I'm very tight on budget. A university student that wants to get rid of his debt and is still learning to cook. I've heard many stories of how people are still in debt from school after decades have passed, and I strive to overcome it as quickly as I possibly can.

Would I be wrong if I don't tip at a restaurant because of my debt?
```

**Verdict:** KEEP

---

## 029 · `12ee9c6710e8b25b` · 1246c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked my neighbours to can it?

```text
Hey y'all, Long time sub lurker here. TL;DR at the bottom. I need advice on the level of douchebaggery I would display if I asked my neighbours to stop singing. For context:  \- the singing is loud enough to be heard with TV on, but not so loud that is heard over headphones. \- our building laws say ppl can make noise within certain hours, so they're not in breach (drats!) \- there are I think at least 2 who are constantly belting to their heart's content. \- hits include: that one part from Beyonce's "Halo" where she repeats higher and higher, Adele and some George Michael? (I mean I like the songs..though not their rendition) \- I think they might be rehearsing because they're only repeating pieces of the song (over and over) with no backup other than the voice. \- Last but not least, most of the times it sounds plain bad. It's like I am forced to be a part of someone's shower concert. Why would I stifle people's creativity? I work from home and have exams coming up. I want to not have to have headphones all the time, in my own home. I feel like I'm one "Hello" away from stomping up there like Mr. Heckles! Thanks for the read! TL;DR: Neighbours sing poorly day in day out.

Would I be wrong if I asked my neighbours to can it?
```

**Verdict:** KEEP

---

## 030 · `130b9bb9a1edda39` · 1435c · ask=body+title

> **Title** (not part of the prompt): WIBTA If I refuse to do a job because I dont like the way the owner wants it done.

```text
I do handyman work, mostly painting, on the side. I have a long time client who happens to be my landlord and is often frustrating to work with. Currently he wants me to paint doors and trim work inside an apartment with tenants currently living there with a two year old. He is notorious for short cuts while always claiming he wants a quality product for his residents. Currently its revolving around a paint for doors and trim that doesn't cover and is a pain in the ass to work with because it sometimes takes three coats to actually cover. It's basically a pre pigmant added paint. Think the stuff they grab and then add pigmant to after you ask for a specific color. 

He and I talked about it after the last job, last week, and he said he would look into getting something different for the next. The next job is Wednesday. He messaged me today and said he's keeping the old paint. I completely feel like I waste my time using this stuff not to mention I will have to work around a two year old when it's already a pain in my ass painting large portions of an apartment with people still living there. Imagine being told you cant use your only bathroom for over 5 hours because it takes two or three coats for the paint on the door to dry.
Would I be an asshole for telling him that since he still using the same stuff I wont do it?

Would I be wrong if I refuse to do a job because I dont like the way the owner wants it done.?
```

**Verdict:** KEEP

---

## 031 · `1316b74ec184472e` · 599c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I called the animal shelter in a already owned dog

```text
So there's a dog who's already owned but the thing she's treated very poorly, she's a sweet lovable dog and dosent deserve what she gets. They never let her inside, they Litterly throw her food at her, and she's always tied to a post and only allowed to move 10 feet in any direction. The family who already owns the dog, had a animal die due to the severe neglect of him. He died after being stuck in his cage so long he couldn't use his legs. So should I call animal control or not? (Where I live there are no kill shelters)

Would I be wrong if I called the animal shelter in a already owned dog?
```

**Verdict:** KEEP

---

## 032 · `1334fb5b5952d079` · 372c · ask=closer

> **Title** (not part of the prompt): WIBTA Selling GPUs without disclosing upfront they were used for crypto mining

```text
So I am selling a couple of GPUs in ebay that were previously used for crypto mining for 6 months ish

I'm thinking of not writing in the post that they were used for mining but if anyone asks I will obviously say they have been.

They are otherwise in good condition, have been cleaned regularly, kept in a cool environment and haven't been overclocked

What should I do?
```

**Verdict:** KEEP

---

## 033 · `1399c003dafbfdd0` · 576c · ask=title

> **Title** (not part of the prompt): WIBTA for not buying a ticket for my friend’s husband and son?

```text
I want to buy tickets to a show for my friend for Christmas. They’re a little bit on the pricey side, which is where I’m stuck. 

I’m happy to buy her a ticket + 1 extra - but she has a kid and partner. Is it a total dick move to not get one for both the spouse (who I’ve never met) and kid? I figure 2 tickets lets her choose which one of them she’d like to bring (date night or kid bonding time) but I can’t really afford three. Or of course she can just ditch them both ;) 

Thoughts? (And thanks!)

Would I be wrong for not buying a ticket for my friend’s husband and son?
```

**Verdict:** KEEP

---

## 034 · `13ccc36d40a07657` · 894c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA, if I stopped contacting a ''friend'' for 2+ months to see if there even is a friendship.

```text
Probably a shitty title. Long story short had this friend who I considered myself close with but for quite some time now our talks are at least in my opinion stale and lifeless and lacking any meaning to them and my ''friend'' has changed a lot too doing a lot of things I don't really can tolerate(Like getting angry at things that never bothered her before) and if she would have been doing these things before I would never have become friends with her. Honestly it also feels like I am putting in 80% of the effort and getting back 20% as I lead all the convos etc start them. Honestly it's a huge pain in the ass and I just flat out just wanna tell the person that this isn't working, but I don't have the heart to go through with the process. You know taking the ''easy'' way out?

Would I be wrong if I stopped contacting a ''friend'' for 2+ months to see if there even is a friendship.?
```

**Verdict:** KEEP

---

## 035 · `14299e7be3d8f2a5` · 508c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told my new coworker to stop slurping their coffee

```text
Recently my office rearranged where a lot of people sit. We are all in cubicles and the person that moved in adjacent to me is a slurper. Every morning for around two hours "Terry" will drink coffee and slurp Every. single. Sip. I absolutely hate it when people slurp beverages because it makes my skin crawl and it is completely unnecessary. This is not the slurping of soup which is normal and often needed- this is only beverages.

Would I be wrong if I told my new coworker to stop slurping their coffee?
```

**Verdict:** KEEP

---

## 036 · `1489ca7c6074fbb7` · 597c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if i borrowed my brothers coat from time to time?

```text
Today me and my brother (im adopted into the family) went to the VANS store to buy the new david bowie shoes, however we instead walked out with a coat and matching shoes both for my brother, we both love the coat and i asked him if i could buy of my own when i have the money and he said he wouldnt want to be matching which is fair enough. But he did say i can borrow it when needed because i dont have a coat at all. When we got home to show my adopted parents my adopted mother said she doesnt want to see me borrow the coat.

Would I be wrong if i borrowed my brothers coat from time to time?
```

**Verdict:** KEEP

---

## 037 · `14a8f3a7bcd76459` · 624c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I wouldn't pay for the repair

```text
So, this will be a short one. Obligatory mobile and not first language excuses. My laptop started acting up around 13 months ago. I took it to a local repair shop and it has been working fine until last week. So yesterday I went to the same shop. They've checked it. I mentioned, that I've been here with it before. He checked and said it will be repaired under warranty, which was over 6 months ago. I even asked him if it would really be free and under warranty, he said yes without checking  I'm a broke college student so avoiding a hefty repair cost would be awesome.

Would I be wrong if I wouldn't pay for the repair?
```

**Verdict:** KEEP

---

## 038 · `14f582ee542f3f9c` · 1099c · ask=title

> **Title** (not part of the prompt): WIBTA If I kicked out a drug addict from our house?

```text
So, It goes a little like this. My roommate and I on the lease, 

have taken in his "friend" who just so happens to be a drug addict in.

She's constantly bringing home drugs, and doing them inside our home

(which is smoke free). I have shared my concerns about this with him. 

While he sits there and does nothing. 

I have suspicions that he's partaking in these drugs with her.

Also, she contributes nothing in terms of financial aid, 

whilst eating food I buy without replacing squat.

Being loud, sneaking her crack buddies in to stay for a few nights,

without consulting me or my roomie for that matter.

So I'd love nothing more to kick her out.

On the other hand, here in Central Alberta,

We're currently getting extreme cold warnings, as weather goes. 

So it wouldn't be right to send her on her way.

However, there are plenty of shelters for this type of situation of hers.

Think I'm going to call the landlord, and get rid of her completely.

If she freezes to death, she freezes to death. Not my problem, anyways.

Would I be wrong if I kicked out a drug addict from our house?
```

**Verdict:** KEEP

---

## 039 · `14fc8337c2f3a544` · 970c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I left immediately after arriving at my parents house because my mom won't get off the phone?

```text
Tbh, the title pretty much says it all. Almost every time I'm visiting my mother(maybe once or twice a week), my sister calls. My mom will always drop whatever conversation we're having to answer the phone and will proceed to make me sit there and wait sometimes upwards of thirty minutes. She doesn't make any effort to politely reschedule the call. It'd be one thing if this happened once in a while but it's quite genuinely every time I'm over. It's even become a bit of a running joke. Well, I'm sitting in her kitchen right now, after coming over to pick up some food for my sister's baby shower. My mom answers the door and doesn't greet me, and I realize she's on the phone again. I'm just sitting here and have been for 15 minutes and am frustrated because I can't stay long. I know she'd be livid, but it'd be rude interrupt her conversation too.

Would I be wrong if I left immediately after arriving at my parents house because my mom won't get off the phone?
```

**Verdict:** KEEP

---

## 040 · `15cd24c5912fc659` · 1552c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I stole from a store that overcharged me?

```text
Some context: I have never stolen anything, or even asked for a cup of water and then filled it up with soda, or anything like that. But this time I feel like it's maybe justified. I was in a grocery store and felt like treating myself, so when I saw this brand of cookies I really liked on sale for $1.50, I decided to buy them. When I went to pay, I thought my total ended up being a bit high, but the line was so long and everybody had already been waiting for so long that I decided not to hold up the line since it was probably nothing. Then, when I got to my car, I unfortunately forgot to double-check the receipt, which is on me. At home the next day, I found the receipt in my pocket and checked it, then found out that they charged me $4.00. I don't know if the register glitched or something, because below the name of the cookies, it even said that they should have been $1.50, but then the number next to it said $4.00. That is so much to spend on a single box of cookies that were already kind of a splurge. I would let it go if they overcharged me by $.50 or even a dollar, but the thought of paying $2.50 extra is making me feel so guilty. However, the store in question is kind of far away since I had stopped by it on the way back from another place, and is not my normal store, so it would cost me that much in gas to go back and get my money back. The next time I'll be up there is in a couple weeks. Either option is making me feel terrible, so I appreciate any input.

Would I be wrong if I stole from a store that overcharged me?
```

**Verdict:** KEEP

---

## 041 · `1619101852957afa` · 1566c · ask=title

> **Title** (not part of the prompt): WIBTA For avoiding a friend because they are really negative?

```text
So I have been feeling this way for awhile and I feel like it would be helpful to get your input. 

so i have this friend we will call Abby. Me an Abby met at the school I work at and have been friends for about a year but lately im just getting tired of her shit. 

for one Abby judges everybody about everything. it drives me up a wall. One of the things that brought us together was being sassy but she takes it way to far. like we were out to dinner with a mutual friend last night and this friend said she can really handle spicy food. Which I get, abby then made a fb post about how she hates people who hate spicy food.

im sorry but how is that a thing? but she is always judging someone for something and i just (as lame as this is about to sound) do not want all that negative bullshit in my life. 

more over she does this thing where if im talking about my day whether good or bad she clearly doesnt care. She is always looking to talk about herself. If i text her about how a student was getting on my nerves or how i slipped in the parking lot she will give me a laughing face emoji then tell me about her day. and if I dont respond with a paragraph or something she then judges me for not being supportive. Then if i am getting on about something bad that happened to me she will have to one up it. Every time. 

lately she is just driving me up a wall and I am over it. I told my other friend that I was planning to cut ties and she said I was being picky and needed to chill.

Would I be wrong for avoiding a friend because they are really negative?
```

**Verdict:** KEEP

---

## 042 · `169ebf8b58f3b7f7` · 811c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I didn’t invite my housemate’s boyfriend to my birthday drinks thing tonight?

```text
My housemate has a boyfriend who lives pretty far so he’s staying the weekend for Valentine’s Day so they can spend time together. I like him and he’s a very nice chill dude - he cleans after himself and sometimes gives me food if they have leftovers from cooking. 

Today is my birthday and after a very stressful week of midterms and work I decided I want to get fucked up with my friends at the bars. I just want it to be me and my friends (so 4 total) because we rarely get to hang out with us all working and school. I feel like I should invite him bc he’s nice and if I didn’t he’d just be here alone. I haven’t talked to my housemate yet, but would it be rude of me to tell her I don’t want to invite him?

Would I be wrong if I didn’t invite my housemate’s boyfriend to my birthday drinks thing tonight?
```

**Verdict:** KEEP

---

## 043 · `16ac818f33b93e8f` · 633c · ask=title

> **Title** (not part of the prompt): WIBTA If i used my old friends to attract my crash to me?

```text
My loved one said that she lacks contact with ppl and she would like to go with me at my hippielike parties with fire and guitars, but i dont have that much contact with people from those parties and i want to talk to them more now, so i can have a party to invite her to.

 I like them and i think they enjoy my company too, but i just stopped talking to them becouse of the distance and bcs im busy with school. 

Now i just want to meet with her and this would be the best option but i wonder if i am using them too 

(English is my second language disclaimer)

Would I be wrong if i used my old friends to attract my crash to me?
```

**Verdict:** KEEP

---

## 044 · `177e28809012f120` · 1061c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I think I know my friend broke my phone on purpose and I asked him to pay?

```text
So I was at a party about a week ago when my friend took my phone as joke while it was unlocked. He has done this plenty of times before so I thought it was fine but apparently not. He gave the phone back to me and the bottom of my phone is cracked and I'm just there like a muppet like what the fuck. He hands me £15 for the damages but ngl I doubt the screen repair on a S8 is gonna cost £15... We were all talking on gc about it the next day and they said it didnt matter so I thought I'd just let it go because its only a cracked screen. But just yesterday i was talking to  a different friend about it he started laughing and told me he was there and saw what had happened. He told me he dropped my phone then stomped on it? I dont know why tf he would do that but that don't sound like an accident. Should i ask him to pay for the screen repair or would I sound like a dickhead? And why the hell would he stomp on my phone :/ i thought we were friends lmao :(

Would I be wrong if I think I know my friend broke my phone on purpose and I asked him to pay?
```

**Verdict:** KEEP

---

## 045 · `186405fa85cfe739` · 1470c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for asking a socially awkward person to screw off

```text
Let’s call this socially awkward person Tom for simplicity sake. He has difficulties talking to anyone due to the fact that he sees himself above most common people. Whenever he attempts to talk normally to someone, it would usually end up with him offending someone. He also has many other eccentric behaviours like talking to himself during lessons and enjoys showing off his knowledge in front of the class. This results in him having no friends as nobody is willing to deal with his insults or his eccentric behaviour. However, since last year, he has been very clingy on my group of close friends. He has since invited himself to our table everyday during lunch and as nobody is willing to strike up a conversation with him, he spends most of his time glued to his phone. I feel extremely awkward and annoyed by him sitting around with my close group of friends as he is hindering me from having private conversations with my own friends. I have tried giving him many hints that he should just go somewhere else, but to no avail. He always trails my group and this has been going on for over a year already. He has personally insulted me and my friends quite a number of time and I really want him gone as he’s starting to become a nuisance. TL:DR A socially awkward person who thinks that he is too intellectual to talk to others keeps sticking to me and my group of friends and I want him gone.

Would I be wrong for asking a socially awkward person to screw off?
```

**Verdict:** KEEP

---

## 046 · `18f911d6e4cd59bf` · 2150c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for turning the neighbors in for having dogs?

```text
Context: I live in some townhomes that are independently owned, but many are owned and rented out. We rent and the people next door rent as well (from different landlords). The HOA rules state that no dogs are allowed. We got an exception because I have a ten pound, very well-behaved dog. He barks when someone knocks on the door but other than that I doubt anyone in our neighborhood knows he exists. It’s important though that we actually have permission to have a dog. We didn’t sneak him past anyone. Most of the neighbors do have dogs. It isn’t a well kept secret, but it doesn’t bother us. The neighbors directly next door, though, are starting to be a problem. They have two big breed dogs in a ting apartment. They only let them out on to a tiny patio. It’s the kind of dog we want- but refuse to get yet because they deserve (and need)  a massive yard. When the owners aren’t home the dogs whine / groan all day. It’s a sad sound that makes my dog pace. For hours a day. When they come home they put the dogs upstairs and begin band practice. Band practice is another issue. For a townhome with thin walls I don’t know why they’ve decided to practice here every night. It’s gotten louder as time goes on. At first, we were just like “well they pay to live here too”. But then it started going later and later and getting louder and louder. To the point where we can’t watch tv because we can’t hear it. We’ve asked them to turn it down twice and they’ve been very nice and complied. But I don’t think we should *have* to ask. During band practice the dogs are even more worked up, too. They know people are home so they bark to hang out. And then get yelled at to shut up. The yelling makes it even worse. I can’t really drown it out. White noise isn’t enough unless I’m wearing headphones. I shouldn’t have to wear those 24/7 in my own home though. I know it’d result in them violating the lease and being made to move immediately (or sadly giving the dogs up, which I don’t want- but they are not being cared for) while complaining about the music would be a slower process.

Would I be wrong for turning the neighbors in for having dogs?
```

**Verdict:** KEEP

---

## 047 · `1906458ff4561f39` · 890c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for making my coworker do his share of inventory even though he has a broken foot?

```text
The way my company works is coworkers opt in for a shift and then work that shift. If you opt in for an open, you have opening duties. Part of our opening duties require us to scan inventory in daily around our location, and that requires a bit of crouching and walking around. Certainly not the easiest thing to do with a broken foot. My coworker recently broke his foot and has been slowly using that as an excuse to not do his duties. The issue I have is that he's choosing to work open shifts, and there are other shifts scattered throughout the day he could switch other coworkers with. I feel like if you're going to volunteer yourself for an open shift that you chose, that you should be required to do open duties. If you're unfit to do the duties, then don't select an open shift.

Would I be wrong for making my coworker do his share of inventory even though he has a broken foot?
```

**Verdict:** KEEP

---

## 048 · `198928283dee03f8` · 1084c · ask=body+title

> **Title** (not part of the prompt): WIBTA for not getting a friend stay with me even though i already said they could?

```text
It’s my 21st birthday on Monday and I’m having a party. There is a person in my friend group, let’s call him Dave. I’m not as good friends as I used to be with him because I learned that he was really into Jordan Peterson over the summer and I realized that I didn’t really like spending time with him, he was just a bit annoying. We’re all still friends with the same friends but we don’t really talk anymore. 

I put into our group chat inviting everyone to my party. I have no problem with him coming. But he texted me saying asking if he could stay the night at my place that night. Because I have an overwhelming need to please everyone I said no bother. But I’m thinking now, I don’t really want him to stay at my house. The thought kind of makes me uncomfortable. 

He lives 40 minutes away so he wouldn’t be able to drive home at the end of the night. He might be able to stay with someone else though. 

Would I be an asshole if I took back my offer of letting him stay the night?

Would I be wrong for not getting a friend stay with me even though i already said they could?
```

**Verdict:** KEEP

---

## 049 · `19a8390959c13186` · 967c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I told 2 friends they need to get jobs?

```text
So my group of friends consists of me and 4 other people. 2 of those friends are brothers. The are almost 30 and have never had a job. They still live with their dad in a camper on their family's property. Their dad is in his late 60's and recently was let go from his job. We have a game night every Friday. They are there every Friday and we always get food and basically just play video games and bullshit for a while. They never pitch in for the food of course or they will bum money from their dad to pitch in, which none of us are comfortable taking. So they ask one of our other friends whether his job was hiring currently, and he said he would look into it, thinking they were finally going to step up. However, they were asking because in their words, Their dad needs to get off his lazy ass and get another job soon. Game night is always at my house. Sorry if this is a bit jumbled I'm on mobile

Would I be wrong if I told 2 friends they need to get jobs?
```

**Verdict:** KEEP

---

## 050 · `19f52c9521a2c08a` · 1419c · ask=title

> **Title** (not part of the prompt): WIBTA if I tell an awkward coworker he’s coming on too strong?

```text
I have a coworker that I genuinely think is a good guy, but I don’t think he realizes how he’s coming off.  Once he gets started on a topic he’s interested in he will follow you around and go on and on about it.  He also says inappropriate stuff sometimes.  

The reason I want to address his behavior now though is because he’s making another coworker visibly uncomfortable.  The other coworker is very shy.  I’ve heard her speak only a handful of times besides when she’s dealing with customers.   Today he was talking about video game to her and doing his follow her around thing.  She wasn’t looking at him or responding.  I saw her walk all the way to the back of the store, then turn around and walk all the way to the front without doing anything.  It seemed obvious that she was just trying to get away from him, but he seemed oblivious.  

When she walked back to the front I tried to pull his attention to me by making a comment about a video game I like, but he just said that he didn’t play it and right back to her.  

I don’t think he means any harm, that he just doesn’t read social cues very well.  I know the other employee is an adult and it’s not really my place to speak for her and maybe I should just mind my own business.  Even if I did decide to pull him aside I have no idea how to word it without hurting his feelings.

Would I be wrong if I tell an awkward coworker he’s coming on too strong?
```

**Verdict:** KEEP

---

## 051 · `1cfb307fc3177ab4` · 1299c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for trying to tell one of my peers to stop talking to my friend group

```text
Sorry for bad layout on mobile I have someone in my year let's call him Tom   For background the school I moved to had noone I knew at it so I found it hard to fit in but I eventually got into a group of people with about 4 but sometimes tom would join in at first I had no problems with him but he started to do stuff like sing songs he knew me and my friends hated but you know I didn't mind he then started to do things like make fun of my height I'm pretty short compared to other people he also can't take a joke at all if you say anything not nice even as a joke to him he will immediately threaten to go to the princibale he is know for telling on everyone for everything I've been told by other people that he has asked them weird stuff like for nudes which I disagree with he is just kind of a weird person he has also started a vlogging channel and uploads to tik tokme and my friends hate these forms of entertainment and have asked hi. To not talk about his follower/sub count which he brings up a lot I asked my mother who is always amazing  and very supportive and she told me to deal with him and that he just wants some friends I told her everything and she told me to deal with him pretty much

Would I be wrong for trying to tell one of my peers to stop talking to my friend group?
```

**Verdict:** KEEP

---

## 052 · `1dc58bdd820fdd5b` · 474c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I ignore this email?

```text
I work as a PhD candidate and coworker X graduated last week. Today some of us got an email by her friends. They said, that they were planning a party and we're all invited because we are X's close friends. I didn't know about that. She's nice and all that but I wouldn't consider her a close friend (I also didn't know that she thinks we're close friends). We've maybe talked 2 or 3 times since I work here (2 years) and that's it.

Would I be wrong if I ignore this email?
```

**Verdict:** KEEP

---

## 053 · `1deb217e143e6e7a` · 636c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked my friend to pay the $250 cleaning fee for the car I rented?

```text
I rented a car and me and some friends used it to go on a little ski trip. One of my friends is a big smoker and it's about a 2 hour drive so he asked me if he could smoke in the car. I said no man. He said common bro. I said alright fine. Fast forward to 2 days later when I return the car and the guy says he smells some smoke and has to charge me the $250 cleaning fee. I said I won't pay it but obviously he charged my card anyway. I tried to charge it back with my bank but unfortunately this is one of the rare cases they can't help me out.

Would I be wrong if I asked my friend to pay the $250 cleaning fee for the car I rented?
```

**Verdict:** KEEP

---

## 054 · `1e8575af568eb806` · 1213c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I stopped my friend from using my netflix subscription because he refuses to pay her share

```text
me and my friends decided to share a netflix account. We've agreed that we would divide it between us and since I brought up the idea it would be under my bank card. A month later, i let everyone know a few days before that it was time for them to send their share. Apart from that one friend, everyone else had sent the money to me via bank transfer.  

This one friend of mine had set her skype on do not disturb and i can almost swear she reads my messages but does not respond to them. Seems like a common thing for her to do. 

We are all paying a small amount each, to use the account. I know its not much but a few of us and myself included are students who are horrible with budgeting. Whilst it's none of my business with what she does with her money. She has a job, lives rent free, goes to college(UK) and plays video games all day. 

If she were to respond to my message with something along the lines of "i cant pay im struggling with my finances atm" I would not press this matter any further. 

I don't know if I'm being petty buts its not fair that the rest of us pay and she doesn't.

Would I be wrong if I stopped my friend from using my netflix subscription because he refuses to pay her share?
```

**Verdict:** KEEP

---

## 055 · `1f3202757f14df32` · 2155c · ask=title

> **Title** (not part of the prompt): WIBTA If I asked my SIL to repay vet costs that I helped her with?

```text
I'll be short and sweet here. My SIL lives next door and is kind of a shitty cat owner. She feeds and shelters them, but emotionally neglects them to the point where one cat ran away and lives with a neighbor across the street now, and her other cat basically moved in with us, but goes home for the odd feeding. Husband and I are big cat people, so we always showed affection to their cat when it came over and thus, the unofficial adoption. To be honest, her cat is kinda annoying and our cat isn't a fan, and I wish he didn't like us so much, but what can you do. 

Anyway, about 6 months ago I noticed her cat loosing weight, despite eating a ton of food at our house and at her house. Deduced it was diabetes. Mentioned to her he was skinny, kept hinting he should see a vet, even weighed him and proved he was loosing weight and still she dragged her feet about taking him to the vet. 

Now they don't make a lot of money, but enough for a vet, and plus don't own pets if you can't take care of them. Anyway she is aggressively cheap and I just knew it was about money so I offered to pay half if she took him to the vet. She immediately agrees, makes an appt, takes him to the vet, they run a bunch of tests and treat him for a few things, to the tune of $400. I pay my half as promised, with no expectations that she would pay me back. It was a straight up gift on my part.

Fast forward a few months and her cat is doing much better. Weight is back on and then some. Still spends 90% of it's time chilling at our house. I mean, he sleeps in our bed. 

Here is where I am thinking of potentially entering asshole territory: her grandma died over the summer, shortly after the vet trip, and she just now received an unexpected inheritance of around $30k. I would never expect her to pay me back normally, but now that she has this big windfall, would I be a jerk if I asked her to kick me back the money I paid for her cat? 

Sorry I lied at the beginning and said this would be short and sweet. Anyway, debating asking for the money, but curious on what you all think.

Would I be wrong if I asked my SIL to repay vet costs that I helped her with?
```

**Verdict:** KEEP

---

## 056 · `1fd54ce89c078304` · 572c · ask=title

> **Title** (not part of the prompt): WIBTA if I refused to help my father take my bed?

```text
For a long time, my parents were divorced. He moved back in for reasons too many to name here. He told me to get rid of my mattress because he was giving me his giant queen sized bed. Okay cool. I helped him move it in and shit. 

Well, not even six months later he's telling me that he's going to kick my mom out and now and take my bed. It is a PAIN IN THE ASS to move. My room is tiny as fuck to begin with. It I knew he would take it back in not even a year, I wouldn't have accepted it in the first place.

Would I be wrong if I refused to help my father take my bed?
```

**Verdict:** KEEP

---

## 057 · `205c1a4e386591ec` · 531c · ask=title

> **Title** (not part of the prompt): WIBTA for asking the lady that sits next to me at work to stop chewing gum

```text
This lady chews like a cow. pops smacks - I put my headphones on and try and drown it out with some bob's burgers while I work, but I can hear her through the headphones this morning. it pisses me off, but I don't want to be a jerk - especially since she was just hired on only a few months ago after leaving her job of teaching 8th graders math. I dont know why that's pertinent, but for some reason it plays a part in my decision to feel bad

Would I be wrong for asking the lady that sits next to me at work to stop chewing gum?
```

**Verdict:** KEEP

---

## 058 · `208d1b6fa7bc8dcb` · 954c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I remind my tattoo artist how much he charged me for my last two tattoos upon getting another?

```text
I’m getting my third tattoo from the same artist tomorrow and wasn’t sure if this would be a rude thing to bring up. 

I met with him for a consultation last week and he mentioned that my tattoo would be one to two hundred dollars. It’s going to be about the same size and detailing of my other two tattoos done by him. For the last two he charged only $80 and I left him a tip as well. 

I completely support artists and their pricing, I believe that they typically make it as fair as possible. I just think that if I’ve paid $80 (plus tip) the last two times for very similar size/style tattoos then it should be about the same price. 

I’m not sure what he will finalize the price as yet because each time he told me the price after he was done with it. 

Would this be an appropriate thing to mention or would I sound like an asshole?

Would I be wrong if I remind my tattoo artist how much he charged me for my last two tattoos upon getting another?
```

**Verdict:** KEEP

---

## 059 · `210f8b3da3c8fe58` · 1015c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for not speaking a language

```text
Title is quite vague, but didn't want to go into much detail. Recently, I remembered a family vacation to the south of France we had about 3 years ago, and during that vacation, naturally, we went to experience French cuisine. Now at the time I was somewhat decent in French speaking, but not enough to be fluent in it, but my family had no idea on how to speak French. So, we go to a restaurant, and the waiter comes to take our orders. This is where my dilemma begins. I was too shy to order for my family, so my father decided to do it instead. And instead of butchering an attempt at speaking French, he did, what I thought was the polite thing, and spoke in English. The waiter, however did not like that. He said: "Why do I have to learn English, if you don't bother learning French". Nothing too much out of the ordinary. For some peculiar reason I thought about it today, and didn't know who was in the right, hence my attempt at addressing this to the public.

Would I be wrong for not speaking a language?
```

**Verdict:** KEEP

---

## 060 · `21d5fc20ead895f1` · 2162c · ask=body+title

> **Title** (not part of the prompt): WIBTA for leaving a poor review for a car salesman?

```text
Background: was in a wreck and car was totaled. Husband and I are looking for a car equal to what we're getting back from insurance, but we took out an auto loan so we can get a replacement before our rental runs out. We'll pay it back when we have the payout. We found a car that met all our criteria, but it was two hours away. We called to set up an appointment for Saturday, but he said he expected the car to sell by then. So we made plans to go up yesterday after work. His website shows that he offers financing (important), so I called him back and told him we would be there by the time they close and that we already have financing through our bank so it would be a quick buy. He said okay.

So we drive two hours, look at the car, test drive, talk it over, and decide to buy it. I pull out the financing paperwork and he says he doesn't know how to do it. I told him it's straight forward and that I can email it to him. He said he doesn't work with banks. I pointed out that I told him on the phone that we financed through the bank, and he told me that he didn't hear that part and if he had, he would have told us not to come. So I asked why his website offers financing, and he said that he doesn't actually accept those applications. I asked if he would make an exception since we drove so far, and he said no because he didn't want to do the title work and wait for the money. He then made some snide comment about how it was obvious that we didn't have cash and it would go to someone else who does. So we left.

I can't decide if I want to leave a review for him. On one hand, the car was in good shape and at a very good price. On the other hand, I feel like his website is misleading and he didn't handle the situation well when there was a misunderstanding. He claims he didn't hear me say anything about financing, but I has no reason to believe he dealt in cash only (nothing on his website indicated that).  If I left a review, it would be more of an "FYI" type deal, but I honestly can't decide if that would be helpful/appropriate or if it just makes me a bitter asshole.

Would I be wrong for leaving a poor review for a car salesman?
```

**Verdict:** KEEP

---

## 061 · `22d9e61323d84ef6` · 406c · ask=title

> **Title** (not part of the prompt): WIBTA for leaving early?

```text
I work in an office and literally have nothing to do right now. My direct boss isn't even here and I don't have much freedom to do anything she doesn't dump in my lap. Would it be shitty to go home now even though it's barely 2pm? There could possibly be someone who needs something from me in the next three hours but that stuff usually isn't urgent or even important.

Would I be wrong for leaving early?
```

**Verdict:** KEEP

---

## 062 · `22dd20df0f27908e` · 956c · ask=body+title

> **Title** (not part of the prompt): WIBTA for not giving away my concert ticket?

```text
About a month ago, our school student activities board held a small trivia night for teams of six. The prize for the winning team was a set of six free tickets to a local orchestra concert. I was invited to fill in on a team of six really close friends (all of whom I am also friends with) when one of them was sick and couldn't make it to the event. 

Lo and behold, we end up winning the tickets and the orchestra is tonight. I really want to go but I get the feeling they may want me to offer the ticket to their friend who coudn't make it, as I essentially took his spot. Again, I really want to go, but I didn't really contribute to the success of the team in any meaningful way and I don't want them to dislike me in the future for not giving up the spot. As far as purchasing an extra ticket, we're all college students and none of us can afford one. (Should I maybe split the cost with him?)

Would I be wrong for not giving away my concert ticket?
```

**Verdict:** KEEP

---

## 063 · `237a4ca2070f97a4` · 546c · ask=body+title

> **Title** (not part of the prompt): WIBTA if charge my friends while I use coupons?

```text
Me and a few friends are going on a camping trip soon and I decided to plan it, i had to buy all the food for the trip, i had saved up coupons that equal about  41 dollars that I was going to use to buy my groceries for a week because I like to meal prep. My question is, if I bought everything and the total came out to about $170 should I split it as $170 or as $129 which would be after my coupons. I feel like an asshole if I charge them $170 while I technically payed with coupons.

Would I be wrong if charge my friends while I use coupons?
```

**Verdict:** KEEP

---

## 064 · `242df45e70d0607e` · 1108c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told my apartment building that my roommate has a pet?

```text
I’m in college and live off campus with three other girls. From the get go, two of my roommates bullied me and were mean to me for no reason. They treat me like I don’t live at the apartment and are just generally rude towards me. Living with them is hell and I’m in the process of finding a new place to live for next semester (they don’t know this; side note: we all have separate leases)  I’ve gotten to the point where I’m just kind of like fuck it and have gotten a bit petty because they’re assholes to me no matter what I do. Anyways, so my one crazy roommate bought a gecko without asking anyone. Absolutely no pets are allowed in our apartment building and there is a $1500 fine if they find out you have one. I’m thinking about telling my building that my roommate has a gecko when I move out because she made my life a literal hell this past semester. She’s loaded, so I’m not even sure that $1500 would actually make a dent in her pocket. That’s not why I’d do it though- it’d be more of a final fuck you towards them.

Would I be wrong if I told my apartment building that my roommate has a pet?
```

**Verdict:** KEEP

---

## 065 · `2451d05db7c40450` · 885c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I confront my colleague about our overtime schedule.

```text
This is my first job after graduation. I’m working as a assistant engineer(AE) I’ve become the favourite between the other AE. My supervisor have been giving me overtime now every weekend to help him do some maintenance which gives me double pay that day for a few months now. Recently, one of my co-workers send out a email to my supervisor saying that the overtime schedule is unfair as I’m getting far more days of double pay compared to the rest. My supervisor have then adjusted the schedule to be fairer for the next month. I’m upset about the changes naturally. This make me feel like I’m not welcome here and have make me consider quitting the job. My co-worker has been in the company for many years more than me. I do feel kinda guilty with the schedule with the balance of overtime given before too.

Would I be wrong if I confront my colleague about our overtime schedule.?
```

**Verdict:** KEEP

---

## 066 · `254dee644370be0b` · 854c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I Reported My Neighbor's Parked Car?

```text
I'm from the UK, and any vehicle that is parked on a public road must have a valid MOT as part of the law. The parking situation on our street is pretty dire, and the neighbor has parked their car deliberately on this spot to guarantee that their partner can get another spot right outside of their home. Because of this, I need to park downhill, and more importantly out of the line of sight of my home, possibly invalidating my own insurance. I noticed all of their tyres were flat and checked the tax and MOT status of the vehicle, and it turns out its MOT is invalid. This vehicle has remained here for a month without moving so it looks like they don't have any intent on moving it soon so others can use the spot or to renew their MOT. If I report them they could get a fine of up to £1000.

Would I be wrong if I Reported My Neighbor's Parked Car?
```

**Verdict:** KEEP

---

## 067 · `2565e50f4420bde0` · 939c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I keep quiet about my friend's auto mechanic fees

```text
My friend is teaching me to drive. His car is quite old and had several previous repairs. Last Saturday's lesson was tricky and more than average clutch work, so I stalled more. We were going to pick up another friend and go downtown after the lesson. After picking up our other friend, we were on our way and his clutch wouldn't go back up at one point. We pulled over, he tried solving the problem to no avail. Since my other friend and I couldn't find a permissible parking space near the area we were in, he had to call a tow truck. His car is at the mechanic's and it was a problem with some fluid that was too complicated for him to fix himself, and it might put his car out of commission for good. I messaged him asking what time he'll get his car back on Monday, but no response. The car is his only reliable way to get to work. Money is tight for both of us.

Would I be wrong if I keep quiet about my friend's auto mechanic fees?
```

**Verdict:** KEEP

---

## 068 · `256802e8064b3054` · 550c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I canceled on a dinner date because the person ate a late lunch?

```text
So this has happened more than once already. The first couple of times I didn't think anything of it because she had lunch with her co-workers and it's really not that big a deal. We had desserts and hung out. But today is her birthday and I felt like bringing her out to somewhere nice when she drops this on me again.

I'm just thinking why even plan for a dinner date if you're just going to eat a late lunch and be too full for dinner?

Am I overreacting here?

Would I be wrong if I canceled on a dinner date because the person ate a late lunch?
```

**Verdict:** KEEP

---

## 069 · `25727d52da00a05c` · 966c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I were to tell my dad to stop talking about how much random people like me?

```text
So a little backstory: I have been working during summers in the past 5 years at the local retirement home. This is a small village where almost everyone knows everyone,and of course my dad talks with everyone. During the school year I am away in a city and I dont go home a lot,maybe once a month,but I make sure to talk with my parents once a week. So for the past 5 months my dad talks about how everyone likes me in the village and they talk about me. Honestly I dont want to hear it. I cant believe that adults with their own lives and familys have time to talk about some kid who doesnt even know them. I just dont believe it,but I dont wanna even imply to my dad he is not telling the truth. To be fair its really not like he is pushing it that much,we get through the topic in five senteces usually. Its just getting really annoying talking about it every week.

Would I be wrong if I were to tell my dad to stop talking about how much random people like me?
```

**Verdict:** KEEP

---

## 070 · `26081a238a87dfe6` · 1381c · ask=body+title

> **Title** (not part of the prompt): WIBTA to tell him to stop eating my food?

```text
I have a friend of a friend who has been occasionally staying in my apartment when me or my roommate are away on weekends. Initially he stayed with us on a two week sublet as I was on a work trip, so it worked out perfectly. At that time I told him he was welcome to finish off my food, since I was gone for weeks. He doesn't live in our city, but often visits and needs a place to stay for a few days, and I need someone to feed my cat, so it's been a good mutual arrangement. 

I came back from a recent 5-day trip, exhausted and hungry. He's considerate in taking off the bedding and sheets before he leaves, though the house is a bit messier than it was, but this time I opened the fridge and it was totally empty. He'd cleared out all my food, cheeses, almond butter (that I brought in a suitcase from the US) and I had nothing left. It's been pissing me off, because he doesn't seem to be aware how expensive some of the stuff he's eating is. I'm pretty sparing and I'll make stuff last for ages, and then after one stay of his, it's gone and I have to restock everything.

I'm torn about letting it go and just accepting it as a cost -- he did take care of my cat--or asking him politely in future to just eat the perishable food, but I don't even know how you go about asking that without sounding like a petty asshole?

Would I be wrong to tell him to stop eating my food?
```

**Verdict:** KEEP

---

## 071 · `27373c1496395fd2` · 756c · ask=body+title

> **Title** (not part of the prompt): WIBTA for bring my Muslim family candy that contains pork-based ingredients?

```text
I'm an atheist living in North America. I come from a Muslim background and I'm gonna be visiting my family in the Middle East this Christmas. It would be quite rude to visit without getting them anything. None of them are allergic to pork or anything like that, they simply don't consume it due to religious reasons.  

My family absolutely loves candy and a lot of **really fucking good** candy here in North America that I want them to try contains pork (gelatin). I'm an ex-Muslim myself and I obviously don't think their dietary restrictions are rational. If I never told them, would I be an asshole for letting them consume good candy that I know they'll enjoy?

Would I be wrong for bring my Muslim family candy that contains pork-based ingredients?
```

**Verdict:** KEEP

---

## 072 · `27ba191c4180d62e` · 927c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I dictate where my brothers children go to school (I'm paying for it)?

```text
I am going to be paying for my brothers children's education we luckily are close and have very similar opinions on education. I am prepared to write an education trust fund for his children however I am worried about 3 possible issues that could arise if I don't include a specific institution in the clause. potential issues:

1. He could send them to a lesser institution which does not meet my criteria for what a proper education is. 
2. He could be influenced by his wife to send them to a lesser institution. 
3. He could pocket the money and send them to public school (highly unlikely)

 We have already discussed which schools I would send my own children so i'm not really that worried that he will screw this up however I do know it's best to get things in writing. Would I be overly controlling if I say a specific school?

Would I be wrong if I dictate where my brothers children go to school (I'm paying for it)?
```

**Verdict:** KEEP

---

## 073 · `27cf760968f18051` · 989c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I went out to lunch with an old acquaintance and used a gift card to pay for myself but didn’t offer to pay for theirs even though they know I was just gifted a large amount on the gift card?

```text
I’m going to dinner with a close friend, X, who is still very close to someone who I would no longer consider a friend, but an old acquaintance, Z. Now X is kind of trying to revive my friendship with Z and suggested we all get dinner sometime because they know I just got a gift card to our formerly favorite restaurant to go to as a group. Z offered to smoke us out before eating (maybe as a reciprocation for using the gift card? I can’t really be sure), but I haven’t been smoking recently, so I declined the offer and instead just opted to meet them at the restaurant. Tl;dr acquaintance offered to smoke me out (maybe? Idk how these social cues work) in exchange for using my gift card at a restaurant together, but I don’t smoke. Can I just use the gift card on myself?

Would I be wrong if I went out to lunch with an old acquaintance and used a gift card to pay for myself but didn’t offer to pay for theirs even though they know I was just gifted a large amount on the gift card?
```

**Verdict:** KEEP

---

## 074 · `28004d791f3e90a9` · 2197c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for calling out my friend for what I consider to be really disrespectful behaviour?

```text
One of my roommates is also one of my best friends. I love her to pieces, but she sometimes acts in a way that I find to be incredibly disrespectful and upsetting. She has a really bad habit of making plans, and then at the time of the plans she’ll be doing something else, and not responding to my texts, essentially leaving me hanging. Now this doesn’t happen very often, but often enough that it really gets under my skin. I know that description is kinda vague, so I’ll give a specific example. Last night when she got home we talked and made plans to go to the library today. I said I was going to get up and come early in the day, and she said that she had a dance practice until 2 and would meet me after her practice. So I went to the library, and at about 1:30 I saved her a spot and sent her a text basically saying “hey, I’m at (X library) if you’re still coming after dance”. By 3, she hasn’t texted me back or arrived at the library. We have each other on find my friends, so I checked where she was, thinking maybe her practice went late, and saw that she’s at her friend’s apartment (they’re mutual friends, I’ve been there a bunch and recognized the location). I don’t have any issue with the fact that she went to her friend’s place, but I find it incredibly disrespectful that she didn’t bother to let me know she wasn’t coming when we had plans and I was saving her a seat, especially since this is far from the first time she’s disappeared with these friends when we were supposed to do something. She’s one of my best friends plus we live together and I really don’t want to cause drama if it’s unnecessary, but I also think it’s super rude to not let me know. It would take all of 30 seconds to send me a text saying “hey sorry, I’m actually hanging out with (X) after practice”, but it just feels like she doesn’t care enough to let me know. This is not the first time she’s done something like this, where at the time of our plans she’s not there and not texting me back, and I just feel like she thinks it’s not important to let me know, but maybe I’m just overreacting?

Would I be wrong for calling out my friend for what I consider to be really disrespectful behaviour?
```

**Verdict:** KEEP

---

## 075 · `28a890cea5e3a678` · 608c · ask=closer · scrubbed

> **Title** (not part of the prompt): WIBTA? Someone is asking me uncomfortable questions and requests...

```text
A friend of mine (Malaysian, I'm British) has asked me to help a friend with a project. I've had no "introduction" to this friend, other than a quick hello, and now the friend (call then Fr) is asking me who I am, where I'm from, what I work as... Without introducing themselves or saying what the project is specifically about. All I've been told is that it will be a series of recorded video calls for a project and I'm to "act in a Malaysian style" (I'm sure acting as another ethnicity is not a good idea), and they will coach me to do so. I might be overreacting, so tell me if I am.

What would you do?
```

**Verdict:** KEEP

---

## 076 · `29517b0825ee08e0` · 1264c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for selling concert tickets that I promised a friend?

```text
Recently, a relatively large band that both me and my friend (let’s say Josh) really like has a series of very small shows running nearby to us.We agreed in advance that if I were to get tickets on our we would go together. Instead of paying me for his ticket, Josh and I agreed that we would write off £30 that I owed him (which was approximately the price of the ticket I’d be buying) and we’d call it even. The day of the ticket sale came and I was fortunate enough to get two for the price of £70, approximately £35 each. We were both obviously very happy. A couple of hours later I checked the prices of the tickets and was shocked to see prices as high as £350 selling on resale websites. Obviously, this is a lot of money and has made me consider either selling one or both tickets. Considering my friend had not actually purchased the ticket if I were to sell both tickets and pay him the £30 that I owe him would that make the the asshole? In whatever circumstance, I would pay him the money I owe him initially  however am I the asshole if I do not pay him the full value of his ticket after being sold, which is likely to be around £300 and just the money that he is actually owed?

Would I be wrong for selling concert tickets that I promised a friend?
```

**Verdict:** KEEP

---

## 077 · `2a2acaeeb2ec9a72` · 1502c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I cut someone off because they seem to be involved in an MLM?

```text
Usually this is something I’m 100% comfortable with, but this is weird for me. I met a girl at work the other day that I REALLY liked. Kids around the same age as mine. Really sweet woman. We started talking about one of our common interests and she invited me out for coffee and I was elated. I’m a social person but it’s a little hard for me to make connections like that organically. 

Today she messaged me asking me to post something for an MLM on my wall for her...

I am beyond a little turned off by MLMs. I seriously fucking loathe them to the point that being the crazy MLM hater is a part of my identity. The friends I have know at this point not to even MENTION it to me. But she doesn’t know that, so it’s not as if she was totally oblivious. 

I can’t decide if I’m being too harsh. I hate to be so blunt but it’s easy for me to write off people I’m kind of friends with when they make a choice that they know, thanks to me lol, is unethical and exploitative and still have the balls to ask me to support it. Because they’ve taken what they know about me and ignored it. But this girl has no idea how deeply the hate flows through me, and she could be really nice...

I feel like I’m not going to regret keeping another hun out of my life but my passionate hatred has been the subject of many debates among my friends and family and I just gotta hear it straight- am I being kinda an unreasonable asshole?

Would I be wrong if I cut someone off because they seem to be involved in an MLM?
```

**Verdict:** KEEP

---

## 078 · `2a4ec1da63a39aa0` · 2039c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I declined being Maid of Honour for my best friend of 25 years?

```text
My best friend is getting married and asked me to be her maid of honour. I think it's very sweet that she asked but I don't want to. I could, in theory, afford to, but it's not really something I believe in spending money on. She's very traditional and therefore expects the full monty of traditional wedding costs. If the shoe was on the other foot, it's not a cost I'd ever ask of others. I also live on the other side of the country from her. The approximate budget would be around $4000 as she's requested that I make one trip out for any dress/shoes/accessory related stuff, plus a trip for the wedding itself. It includes paying for a cat sitter, cabs, bachelorette party, wedding gift, etc. Again, I could fairly easily cover that $4000 but part of that is because I'm much more financially frugal than she and most people are. I have a budget, I invest, etc. I live alone in an expensive west coast city and I still have more money left over at the end of the day than her two income household. I kind of resent the assumption she has that I'll just say yes "because obviously I have the money." But it's because I've been more responsible than her. Furthermore, she isn't owed the money. But aside from the money, I am just not into any kind of event planning. Especially since I'm 3 time zones over from her. I can't deal with this kind of stuff at work - it'd be unprofessional - and by the time I get home from work it's too late to make phone calls to book various things. That said, she's my best friend in the whole world and part of me thinks that I should just eat the $4k so that our friendship can continue. I don't think we'd \*not\* be friends if I declined, but the idea of her feeling unsupported by me really upsets me, too. I love her. She's family. TL;DR My best friend (31F) asked me (31F) to be her MoH in a tradtional and expensive wedding. I don't want to spend the money nor put in the effort. These are not things I value.

Would I be wrong if I declined being Maid of Honour for my best friend of 25 years?
```

**Verdict:** KEEP

---

## 079 · `2a82b87a407e7e73` · 1699c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for getting a thermostat that locks my roommate out?

```text
So me(24/F) and my roommate, let's call him G(27/m), have recently moved in together. We were not very good friends beforehand- but since we moved in together, he has been trying to be my best friend. I state this because it feels like it makes him feel like he can get away with more things and cross more boundaries this way. However, the major issue at hand for me is financial things. He eats my food, and justifies it by saying he made a meal for us- when I didn't ask for it. I don't have that kind of money as someone who's out on her own for the first time. The MAJOR issue is that he insists on turning the heat up to 80, when we pay gas separately. We decided on a 50/50 split of all utility bills which was great for me at first. So, I decided to talk to him about this and tell him that 80 degrees especially during springish weather is not financially smart and I would prefer him to keep it lower. He would then go behind my back later, after I turn it down, to turn it to something like "79" so it's not technically 80. So, being at my wit's end, I ask him if we could change the dynamic of the bills considering he sees a higher temperature as something worth it to him- whereas for me, that high of degrees honestly has me sweating. He says no. He says that in the adult world, that's one of the challenges you deal with as roommates, and he doesn't consider it a "50/50" roommate arrangement if I'm only paying a third of the heating bill. So, I decided I'm going to get a smart thermostat that connects to my phone- and will allow me to lock the temperature from being changed while I'm gone, in my room, etc.

Would I be wrong for getting a thermostat that locks my roommate out?
```

**Verdict:** KEEP

---

## 080 · `2b31ced4a822880d` · 2015c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for not passing along brothers childhood baby clothes to SIL...?

```text
Me and my older brother are both in our 30s, married and have 1 kid each. I have a boy under 2 & hes got a girl who is just shy of starting school. When my neice was born, my mother pulled out the baby clothes from when my brother & i were babies, having saved them specifically for her grandbabies to have. (She bought them for my brother and when i came along, i wore them too.) Sister-in-law wanted nothing to do with them for neice. Most are pretty gender neutral, so when my mom was babysitting and neice needed clothes changed, she would put on some of said clothes (a pair of jeans for example) so she could enjoy the outfit for a while, but SIL would come to pick her up and change her out of clothes before heading home, leaving childhood clothes at my moms house. We never asked wtf but theorized that maybe its because they were too masculine looking. Fast forward a few years and i have my son. My mom gingerly asked if i would like the clothes, expecting more heartbreak over it but i was thrilled & said hell yes. My little guy rocks the clothes pretty frequently & grandma always has a huge grin to see him in them. Now, i found out my brother & SIL are expecting a boy. So I'm stuck with the dilemma of what to do about the baby clothes. I feel im obligated to offer them to her as they WERE me brothers first, but she has a history of refusing to put baby in them. I worry about it because she is definitely a "gift horse in the mouth" sort of person and has told us often of recieving a gift that wasnt the "right brand" so she sold it and put cash towards the "right" one. My concern is if i give her the clothes, whether on not they are worn by future nephew, that she will toss or sell them in the end. If she doesnt want them or they didnt have any more kids, i intended to put them away for my grandbabies to inherit. Or could i offer them and tell her if she doesnt want to keep forever to return them to me?

Would I be wrong for not passing along brothers childhood baby clothes to SIL...?
```

**Verdict:** KEEP

---

## 081 · `2cfe2cb3d60f2300` · 758c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for pressuring my father to give me more video game privileges?

```text
On mobile, first post... blah blah blah. To give context I’ll start at Christmas. I got my first gaming pc and with it a indie game called counter strike global offensive. I got this game without a mic. Never had a mic before. My dad eventually caved in and gave me a mic. Good right? Wrong. My dad bought the game as well to play with me because he could now communicate with me. Unfortunately, on a solo play through, my dad ran into some heavy multiplayer toxicity. It put a bad taste in his mouth about the community. Because of this I was banned from playing csgo without my dad playing with me. Keep in mind I had just gotten the pc and didn’t have a large game library.

Would I be wrong for pressuring my father to give me more video game privileges?
```

**Verdict:** KEEP

---

## 082 · `2d62054202e35aed` · 1138c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I refused to pay my expected share of utilities this month?

```text
So I got a text from my live-in landlord (who is also a good friend) that our smud bill is $268 right now (up from approx $120), most likely because of the usage of the heaters. She said that she is nixing heater usage, including personal heaters and heated blankets. I don't use heating equipment; if I get cold I wrap myself up in a blanket. I do have a personal heater that my landlord usually uses in her room and sometimes sets up in my room when we're hanging out. We have another roommate, Reese, who I know uses the heater and forgets to turn it off. I have made mention of times when I come home and no one else is home, and the heater is still on. Even texting her right now, she's confirming that she comes home and it's on because "he forgot to turn it off" I really feel like I'm being taken advantage of if I'm expected to foot this bill. Most of the time, I'm not even home when it's on, so I'm not getting the benefits of the heat regardless of my decision. So I really feel like telling my landlord that this spike is between her and Reese.

Would I be wrong if I refused to pay my expected share of utilities this month?
```

**Verdict:** KEEP

---

## 083 · `2d68d0880cc78c89` · 2207c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I sent this email?

```text
So context: I recently complained to DoorDash that I didn't receive all the items I'd paid for. I've had to do this a few times lately, so they put it under review. I just received this email: " Thank you for reaching out and providing feedback on your recent delivery. Unfortunately, we are unable to provide any compensation based on the information you provided and the details of your order. Your reference number for this request is \[redacted\]" I also have a friend who works as a dasher for them and is paid about 3 bucks an hour on a good day because she's forced to waste her time and gas to find orders (I know orders are limited, but InstaCart requires you to sign up for a certain time in advance so they don't have too many people, so there are obviously better ways to handle the situation) Now, I desperately want to send the following email. "Whoever reads this, if anyone does, my anger is not directed at you, it’s directed at the institution. I know these complaints will go nowhere but I’m getting fed up and would like to at least pretend that I can make my voice heard. I apologize for any cruelty that may come across in the following message:  I bought a pizza. I was not given a pizza. I bought a dessert that came with ice cream. I did not receive ice cream. I don’t understand why that’s not enough information. I really liked your service when I first discovered it a couple years ago. Now, it seems as though the service quality has totally dropped, you pay your dashers slave wages without even compensation for gas, you even seem to take their tips from what I’ve heard from my friends who are drivers, and you increase the prices of dishes to get more money, while still taking advantage of the dashers who actually do the work and provide the service. I’ve recommended the service over competitors and used it despite obscenely high delivery fees but you can be assured that won’t be happening any longer. I am extremely disappointed in your management. If you could send this to anyone who might actually care and be able to do something to rectify this repugnant behavior, I would appreciate it. Sincerely, A dissatisfied customer"

Would I be wrong if I sent this email?
```

**Verdict:** KEEP

---

## 084 · `2ff78d95098eee03` · 857c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for reporting my neighbor's new building to the county?

```text
Empty lot next to me for literally ever. New guy buys it and starts cutting down trees and pouring concrete pads, etc. IDK if he has permits for anything but it always seems like he and his buddies doing the work, not *real* workers. I want to call the county and give them a heads-up, so they can check if he has permits, then whatever they do if not (fines, tear it out, etc). I will admit I'm bitter about not having an empty, wooded lot next to me anymore. Also, before he poured a driveway skirt, he was using my driveway without asking as a way to get on to his property. And knocked down my mailbox and didn't fix/replace it. I realize I have to live next door to this guy for a long time, but the relationship did not start off well and I'm not very good at Bury The Hatchet.

Would I be wrong for reporting my neighbor's new building to the county?
```

**Verdict:** KEEP

---

## 085 · `3041c44bca463406` · 1093c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for complaining about my raise at work?

```text
So, I (21F) recently had my one year anniversary working at a daycare. My boss called me in for my evaluation on Thursday, and it went very well. She told me how much they value me as and employee and said I was one of the best (not trying to brag, I think it's applicable to the situation). She then informed me that I'd be getting a raise, I just need to sign the paperwork. The paper she gave me said that my new rate of pay would be $11.30. I was so happy because I am currently being paid $10.50. Eighty cents is a good raise. However I then noticed that she had made a mistake and put that I am currently being paid $11. I pointed out the mistake. She seemed very surprised and confused. She had thought I was being paid $11 this entire time. She told me she'd fix everything and sent me back to work. Friday night, she texts me saying that she had fixed the paperwork and my new rate of pay would be $11. I just need to sign it on Monday. I understand there was a mixup, but I already feel very underpaid for the work that I do.

Would I be wrong for complaining about my raise at work?
```

**Verdict:** KEEP

---

## 086 · `30ef727ddfa92189` · 1049c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I don’t agree to a dog with my roommate?

```text
I am planning to move in with a friend in a few months and she has been talking about us getting a puppy. While I am perfectly okay with her getting a puppy, I don’t want to be legally or financially liable for the dog. I just finished college and am not planning to stay in our current city for more than a few years. I think it’s irresponsible of me to agree to a dog when I don’t know where I’m going to be in a few years and don’t want to hinder my future because of a dog. I also know she has no idea how to train a dog and she has mentioned that I would be the main one taking the time to train the puppy. That’s 6-12 months of my life cleaning up pee and poop inside my apartment. Obviously, I will help with the dog, as in feeding them and taking them out when I’m home. But I don’t want to commit to something and then when I move out, we would have to decide who gets the dog. I haven’t told her how I feel other than let’s wait and see how everything is before we get a puppy.

Would I be wrong if I don’t agree to a dog with my roommate?
```

**Verdict:** KEEP

---

## 087 · `310e62a9f7a54689` · 1977c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I snitched on my manager for very shady/fishy practices?

```text
So basically, yesterday was a huge day for our store... it was insanely busy, and a perfect time to hit some quotas for us. However, we were struggling a lot to hit the quota of our memberships. To break it down, my store has two different types of memberships; a lower price, and a higher price (the higher price allows for additional savings, free shipping, money to be earned back by the member, and so forth). If our quota for selling the more expensive memberships isn't hit, we get talked to about our tactics, and we have to sell EVEN MORE. Plus, we don't get a bonus or anything like that at the end of the year. So yesterday, we were doing awful for these types of memberships. So my manager, in the last hour of the store being open, tells a couple of the cashiers, and a supervisor, to offer it to the members for FREE. That's right... free. These memberships cost upwards of $100 a year. And he is giving it to them for free. So I questioned him on it. I asked him "why are you giving them away? And how are you going to make up for the money lost for them?"  And he responded with: "Well we need to hit our quota, and I'll just write them through the system. I'll do a CFT so they won't show up in the money system for the store."  So basically, he is hiding that he is giving these away to members from the boss. These $100 memberships that he is just GIVING AWAY. Besides that, he has been known to pull other shady things (treating customers like shit, reselling used product so that they can add additional warranties and promotions onto the product, giving promotions to his friends and coding them for other areas of the store so that they get more money/hours, stuff like that). I just feel like he is being super shady and greedy, because he knows that if the quota isn't hit... he gets a talking to. I feel my manager has some shady practices, and is hiding them from the boss.

Would I be wrong if I snitched on my manager for very shady/fishy practices?
```

**Verdict:** KEEP

---

## 088 · `31634ca54c93b31e` · 683c · ask=title

> **Title** (not part of the prompt): WIBTA for disobeying my stepfather?

```text
My mom is going on a trip tomorrow.

She had a row with my stepfather just last night.

Stepfather is kind of a control freak and because they aren’t speaking, he tells me to weasel out information about my mother’s trip including the name and contact of her friends, the tour guide, where she is staying etc. 

He didn’t say it out loud but his tone clearly screams ‘Do this using any means necessary’ 

I think he wants me to break into my mom’s phone.

I know that him having this information will make my mom feel unhappy and betrayed.

I just think it’s an invasion of privacy and unfair for him to use me like this. I feel dirty.

Would I be wrong for disobeying my stepfather?
```

**Verdict:** KEEP

---

## 089 · `3199982c219e876c` · 604c · ask=title

> **Title** (not part of the prompt): WIBTA if I skipped out on my family's Christmas party to hang out with a friend?

```text
So Smash Ultimate is coming out tomorrow, and my friend prepurchased it. After school tomorrow he's gonna have me and a bunch of friends over until 11 or so at night. I've never hung out with my friend outside of school before, so I'm really excited. 

However, my family's Christmas party is also tomorrow night. I've never skipped out on it before outside of one time where we got snowed in. My parents said that it would be okay if I stayed over there instead of going, but I still feel bad for skipping out.

Would I be wrong if I skipped out on my family's Christmas party to hang out with a friend?
```

**Verdict:** KEEP

---

## 090 · `31cc39d089dca315` · 1157c · ask=title

> **Title** (not part of the prompt): WIBTA if I call the leasing office on a guy who keeps puking?

```text
TLDR- dude below me vomits multiple time a day; now my room is starting to smell like vomit

So I live in an apartment that has one floor above me and one below me. The walls are EXTREMELY thin, so we all can hear each other all the time. Since my lease started in August, I’ve noticed that I hear the sound of vomiting every morning and every afternoon. It is very obviously vomiting and it happens multiple times a day, every day, without fail. Although it’s a bit disruptive and gross to hear, I’ve never said anything because like, obviously the guy is having a rough enough time as is; I’m sure he doesn’t want to be puking just as much as I don’t want to hear it. 

Lately though, as it’s begun to heat up, my bedroom has begun to smell like puke. You know that smell after you clean up puke but it still lingers? Like that. I figure it’s coming from his bathroom that is probably directly below me. 

Would I be an asshole for calling the leasing office about this? I’m honestly not even sure what they could do besides notify him about the fact that others can hear/smell it.

Would I be wrong if I call the leasing office on a guy who keeps puking?
```

**Verdict:** KEEP

---

## 091 · `3225e98e50d3f01f` · 1127c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I reschedule an appointment two days in a row?

```text
I had an appointment at 2 P.M. yesterday at a lab to go get my teeth shade matched to be sent off to my dentist so he could make me the proper colored fake cap for my tooth. Unfortunately, I had tire issues and called around 10 AM to let them know I apologized and would have to go and get a new tire for my car and couldn’t make it. They cheerfully said “No problem at all!” And I made the appointment for 2 P.M. today instead (I work night shift). Well, odds would be that I would be getting a call from my boss this morning to see if I could come in early. He accepted my request to make me a manager and asked if there’s any way possible I could come in early for training and so he could show me some other things. I’m stoked, but this would cause me to have to cancel said appointment again. It’s 9:30 A.M. currently where I live. I’m not very versed in etiquette when it comes to situations like this. I REALLY want to go early to show my boss I’m serious about this, but feel like it would be extremely rude of my two reschedule the appointment again.

Would I be wrong if I reschedule an appointment two days in a row?
```

**Verdict:** KEEP

---

## 092 · `32a2cf8807459766` · 702c · ask=title

> **Title** (not part of the prompt): WIBTA for requesting to no longer work shifts with an injured coworker?

```text
One of my coworkers broke her leg.  As a result there’s a lot that she can’t do and has to rest frequently.  I really feel for her and her situation, but shifts are a two person job at minimum and it’s causing problems.  My manager has actually reprimanded me because of how late I was getting out of there because I had to do almost all the closing down by myself.  During the actual shift I start running behind right away and it just gets worse throughout.  It sucks this happened to her and I don’t want her to feel bad about her current limits, but I no longer want to work shifts with her until she is healed up.

Would I be wrong for requesting to no longer work shifts with an injured coworker?
```

**Verdict:** KEEP

---

## 093 · `3355167ff66e126e` · 915c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I asked a girl if that's her "natural hair color?"

```text
There's a lady I work with who intrigues me. She has raven colored hair but light colored eyes...I can't remember exactly what color but I think they are hazel or green. I'm a happily married man so I have no intention of hitting on her, although I must admit that I think she's beautiful in a very exotic sort of way. She is obviously American but she seems to have some sort of Mediterranean ancestry if I had to guess. 

Anyway, I can't tell if this is her natural hair color. The hair color is very even, so there's not a hint that it's dyed. But it's possible that she is one of those rare people that have dark hair and light colored eyes.

I was talking to a colleague of mine about this and she says it's rude to ask people if that's their natural hair color. Is that true? If so, what's a more tactful way I can satisfy my curiosity?

Would I be wrong if I asked a girl if that's her "natural hair color?"?
```

**Verdict:** KEEP

---

## 094 · `337eb641a6e3e85e` · 835c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for telling someone I'd watch their luggage and then abandoning it?

```text
I was at the airport today. Got here nice and early like always. My flight boards at 10:05 and I got to my gate at 09:00. Sweet, perfect. Now I can relax and not worry. Two girls come up and sit next to me. They put their bags down and ask if I'm going anywhere. I say no, so they ask if I can watch their bags. Sure, no problem! Happy to help. Five minutes pass, then ten, then 20, then 40. No sign of them anywhere. Finally, after an hour, they come back as I'm about to start boarding (the same flight they are on). Luckily, this worked out. My gate didn't change last minute and I didn't need to use the toilet. I was annoyed, but ultimately it didn't affect my life. They were no where in sight and gave me no indication they'd be gone so long.

Would I be wrong for telling someone I'd watch their luggage and then abandoning it?
```

**Verdict:** KEEP

---

## 095 · `3384ad6de10a8ce0` · 769c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if i told my parents to keep it down?

```text
This happens once, maaaaaaaybe twice a month but i hear it. The dreaded sounds you never wish to hear coming from your parents room. Intercourse. The walls in our house arent sound proof, but the wall between my room and theirs is so thin that i can hear them fluff out/move their duvet on their bed or flick their lightswitch for their bedroom. Let alone.... other sounds. Dont get me wrong, i dont care that they do.... things, but like every other person in the world I DONT WANNA HEAR IT PLEASE. I kinda live with regret because i wont get mad if they do it at a time im supposed to be (but never am) asleep but when they do it when im awake? And they know im awake? Like christ...... Thats when i have issue.

Would I be wrong if i told my parents to keep it down?
```

**Verdict:** KEEP

---

## 096 · `345d70cbae74e831` · 588c · ask=title

> **Title** (not part of the prompt): WIBTA if I didn’t tell a friend they got the answer wrong on a study guide.

```text
I’m doing a study guide for math and I don’t know how to do a problem, so I text my friend she responds telling me how to do the problem. I thank her and go on my way. A few minutes later I check my work and one of the answers (the one she showed me as an example of how to do the problem was wrong.)

I could have checked it wrongly and it could be right, also her tutor apparently showed her this so it should be right, but I feel like I should tell her so she doesn’t use this method on the test.

Would I be wrong if I didn’t tell a friend they got the answer wrong on a study guide.?
```

**Verdict:** KEEP

---

## 097 · `3475d93866a8240f` · 977c · ask=title

> **Title** (not part of the prompt): WIBTA for trying to sleep?

```text
Sorry, I’m on mobile.

So I’m a freshman in college and I’ve never had to share a room before. My roommate is super amazing and I have nothing but one bad thing to say about her. That one bad thing is that she has to sleep with the TV on. We’ve been roommates since August and it’s now November and I haven’t had a good nights sleep since I moved to college. I haven’t said anything to her yet because I didn’t think it was right for me to ask her to sacrifice her sleep for mine, but it’s gotten to the point where I’m close to tears because I’m always so tired. I either can’t fall asleep, wake up multiple times a night, or get woken up at four in the morning all because of her TV.

I’ve tried everything, I’ve tried sleeping pills every night, sleep masks, ear buds, and any combination of the three and I always wake without fail. I really want to bring it up to her but I don’t know if it’s unfair for me to ask her to turn it off.

Would I be wrong for trying to sleep?
```

**Verdict:** KEEP

---

## 098 · `34a2b414ffd3d221` · 838c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I rescinded my invitation to take my roommate to the airport?

```text
My roommate flew across the country today, and last night he asked if I could pick him up on Saturday at 11 am. I said I was pretty sure I can and he took that as a yes, which was all fine. Then today he texted me that he changed his return flight to get In at 9 pm instead of 11 am. The airport is 30 mins from my house, so a bit over an hour round-trip, and it also happens to be right next to my work so I make the drive every weekday. When it was in the morning I didn't care cause I wasn't going to be doing anything on a Saturday morning anyway, but since it's now at night time, even though I don't necessarily have any plans yet I was hoping to do something that night with friends since it is my last Saturday before the semester starts up again.

Would I be wrong if I rescinded my invitation to take my roommate to the airport?
```

**Verdict:** KEEP

---

## 099 · `34b590a3a0fb4552` · 1653c · ask=title

> **Title** (not part of the prompt): WIBTA if I stopped doing errands I personally didn't sign up for?

```text
(Sorry for possible mistakes, English is my second language and I'm on phone)

So, me and my family live in a town on the coast of Mexico and we rent a room to a foreign lady and her dog(she comes, stays 6 months here and goes), we've been doing this for at least 10 years now, everything was fine the first times she came here.

Now when she first came I had no idea my dad, who is friends with her son(he has a bar in town) signed me up as her bell boy, everytime she arrives I take her luggage to her room(her room is on a third floor so I have to go up and down the stairs a couple times to get everything), sometimes go buy stuff she needs, order take out for her, even help her way up the stairs when she comes home wasted.

I'm a big guy, and can handle heavy things and all but I never agreed to do this, I simply do it for courtesy and to not feel like an ass to my dad.

Well the problem I had became bigger the last three years she came, she always arrived unannounced, she just knocked on our door and shoved her stuff on me to take to her room, coming in like she owned the place, shouting at us from her room everytime anything failed(our house is old and sometimes the plumbing acts up).

I personally started really disliking this lady, but I don't want to be seen as an asshole if I stopped helping her, she sometimes brings us gifts and pays me when I help her, but I never said I would be her personal assistant, I also have a life of my own.

TLDR: dad signed me up to be the personal helper of a lady we rent to, after countless stupid events I want out.

Would I be wrong if I stopped doing errands I personally didn't sign up for?
```

**Verdict:** KEEP

---

## 100 · `350548550a082854` · 373c · ask=title

> **Title** (not part of the prompt): WIBTA for keeping $100 I found at work?

```text
I work at a gas station, just found a $100 bill on the ground. Been about 6 hours now and no one’s come for it.

Very tempted to just pocket it if I don’t hear anything by the end of my shift. Although I feel like the right thing to do is leave it here with a note in case someone comes for it in the next couple of days.

Would I be wrong for keeping $100 I found at work?
```

**Verdict:** KEEP

---

## 101 · `36f9821974e9aa1b` · 824c · ask=title

> **Title** (not part of the prompt): WIBTA if I refuse to argue with my project partner

```text
Since I did everything and he's done next to nothing I do see any way I am responsible for his personal failure.

He is currently trying to make me the culprit for him being useless. Mostly I do too much and dont let him room for work ( which is untrue because I give him tasks every week that never get done so they have to be done at some point ). And also I didnt help him setup work environment. This guy struggled to install linux for 2 months. Shit dude ever heard of Google? 

But yeah i think of the worst case scenario and i think that he knows that he's 100% guilty and now he's being a psychopath and manipulate me into thinking at my fault and continue to do the work for him

Also I dropped him out of the project after the argument without notice

Would I be wrong if I refuse to argue with my project partner?
```

**Verdict:** KEEP

---

## 102 · `3752ad8d80a0b89e` · 1041c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I didn't go in to work tomorrow?

```text
So a few weeks ago I went to a concert and had requested off 2 months in advanced and came into work the day before the concert only to find out I didn't work that day but the next day (the night of the concert). It ended up working out though because the coworker who was supposed to work that night was sick so we just switched days. But for the last three weeks I haven't worked a saturday. I made plans to go to a party tomorrow and was sent a picture of the schedule by a friend only to find out I work tomorrow. I told the manager I couldn't come in as I had already made plans but was told since I didn't request off I had to find someone and if not I have to go in. I feel like I shouldn't have to request off on a day I don't work. I wasn't given a heads up that the person had requested the weekend off, and am sure that they called out last minute. Typically when the schedule is changed they have given me a heads up. I've already missed a concert due to a similar situation.

Would I be wrong if I didn't go in to work tomorrow?
```

**Verdict:** KEEP

---

## 103 · `37be4ef16c66fd20` · 854c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I don’t tell my boss he overpaid me?

```text
I work a part time job at a front desk, mainly to help out, I don’t get paid much, basically the last week I maybe worked 7 hours, my boss continues to call me off of work. My boss really doesn’t like me, not sure why, other then he’s in a lot of stress and I was hired by his old GM and he had to fire that GM for numerous problems. As far as I know he doesn’t particularly love anyone he’s hired, I’ve done everything I can to make sure I’m being trained the way boss wants versus what the old GM wanted. I really enjoyed the job for awhile, it was social, and it wasn’t too bad. But lately every time I’m scheduled I get a call saying I’m not needed. I checked my direct deposit this morning and it was almost $150 over what I thought it should be. So am I the asshole if I don’t say anything?

Would I be wrong if I don’t tell my boss he overpaid me?
```

**Verdict:** KEEP

---

## 104 · `37c313932b375be2` · 626c · ask=title

> **Title** (not part of the prompt): WIBTA if I dated someone mainly because they gave me stability.

```text
I don’t actually have anyone in my mind right now I’m just trying to figure out things. 

All the relationships I’ve had in the past have been unstable messes for one reason or another. At this point I just want someone who can give me a stable relationship. To me it doesn’t matter if I’m actually deeply in love with them or find them all that attractive as long as they can provide me with a stable healthy relationship.

But I feel like having that mindset is disingenuous and I would be an asshole to the other person if that were my main goal.

Would I be wrong if I dated someone mainly because they gave me stability.?
```

**Verdict:** KEEP

---

## 105 · `37caa39a408163d6` · 1033c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I had said this to my waiter last night?

```text
Last night my dad and I ate at a local Mexican restaurant we like. They were a little more crowded than usual. I ordered one type of quesadilla (steak) and when they brought out our food, they gave me something different than what I ordered. It was something like a cheeseburger quesadilla (hamburger patty with two types of cheese, lettuce, tomatoes, sour cream) and a side of fries. I didn’t want to say anything so I took a bite and *oh my god*, that was the best quesadilla I ever had! The meat was juicy and 100 times better than the actual one I ordered. I asked my dad if I could tell the waiter, the next time he came by, “this isn’t what I ordered but I am very glad you brought this out because it is delicious! I would never have thought to order this but I know the next time I come by I’m ordering this again! So good!” My dad told me not to say it as it points out they made a mistake on my order, so I didn’t say it. If I did, would I have been an asshole?

Would I be wrong if I had said this to my waiter last night?
```

**Verdict:** KEEP

---

## 106 · `3867d76c23a678ed` · 1094c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked my mom to pay for a missing item in a parcel she signed for?

```text
I was waiting for a parcel to arrive but had to be at work early today. So I asked my mom to sign for it when it comes (I live at home), which we do often for each other, so there’s usually no problem. But she just texted me that she signed for it and only then saw that it was torn open, after the mailman had left. I asked her to send me a picture and one side of box is literally wide open (not sure how she missed that). She sent a photo of the contents and lo and behold something is actually missing. It’s a shaker bottle for supplements that were in the box. I’m pretty annoyed now and she offered to give me her old shaker which is similar, but not the same and much smaller. I declined. The shaker was 10€. Shipping would be 4€ but there’s actually a coupon for free shipping for the next order in the parcel. I actually already emailed the shop, but I think it’s unlikely they’re just gonna send me a free replacement when the parcel was signed for. I’m waiting on their answer. TL;DR see title

Would I be wrong if I asked my mom to pay for a missing item in a parcel she signed for?
```

**Verdict:** KEEP

---

## 107 · `387d6eca3d9a26e9` · 575c · ask=closer

> **Title** (not part of the prompt): WIBTA Am I too blunt?

```text
A couple days ago I went to my grandpa's funeral and while I was there I realized that I an blunt with my feelings, I don't really cry when I get the kind of sad death in the family will bring. I just say ok to the stuff that usually  people cry of. I'm worry that I seem (or am) being disrespectful for when someone is telling me something that is sad becauseI I just say in response is "Oh, ok". I do feel sad but I just can't express that feeling. I don't know I'm having trouble explaining this in words. Sorry for the bad grammar and thanks for reading.

Help me decide.
```

**Verdict:** KEEP

---

## 108 · `38cd526819da8c00` · 1798c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I don’t go on this trip with my mom and sister?

```text
My mom has been planning a trip to Indonesia for months, which I’ve said no to going on every time she’s asked. She keeps putting off buying the plane tickets for it, in the hopes that I’ll eventually come around, but I have no desire to go and consistently say no. The reason I don’t want to go is because we stayed in Indonesia with her family for 8 months when I was 10, and the memories I associate with it are of puppies dying and my mom’s family treating them badly; people over there don’t like dogs, so they leave them outside, throw sticks and rocks at them when they feel like it, will feed them by throwing a handful of rice and watching them fight over it, etc. The last time we went back, I started crying when I saw the newest litter of puppies, because I knew they would probably be dead once we were gone. The whole thing is a long story, but basically the place just reminds me of bad memories and I don’t like being there. The heat, mosquitos, and lack of indoor plumbing sucks too, but i’d put up with that if it were somewhere other than Indonesia. My mom and I didn’t have a great relationship when I was growing up, but we’ve been on better terms in recent years. My dad has told me that this trip is really important to her and that I should go to make her happy. The last time we went to Indonesia (April 2018), I was forced to go despite saying no over and over, and my dad told me that if I don’t want to go the next time, I can say no and not go. Now that I’ve said no and it’s the next time, both my parents want me to go anyway. I don’t want my mom to be sad because I don’t want to go with her to see her family, but I also hated going there the last time and would much rather never go there again.

Would I be wrong if I don’t go on this trip with my mom and sister?
```

**Verdict:** KEEP

---

## 109 · `397f3902a0d37d88` · 1346c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for not passing my savings on to my friends for our annual trip?

```text
Every year, my friends and I schedule a trip to catch up and hang out, seeing as we are in different cities. It was decided this year that I would schedule the Airbnb for our stay, to be later reimbursed by everyone. This isn't typically a problem, but not everyone has Venmo/Zelle, so the organizer has to sometimes wait until the start of the trip in order to be paid back in cash. The trip is typically scheduled half a year out. The total for our Airbnb was roughly $800 for our entire stay. Since I had never opened an Airbnb account, I searched for a promotion and found that I could get $40 off my first stay. Later, I also saw a credit card offer in my banking app that would reimburse me 15% in statement credit for any bookings on Airbnb. Finally, when booking the Airbnb, I received 1.5% cash back as part of my credit card's rewards. In total, I am saving roughly $150 by making this purchase on behalf of everyone. I'm a bit conflicted about where the line is between the money that I should keep, and what I am obligated to share. Part of me feels that I'm justified, in that I volunteered to do something no one was particularly enthusiastic about doing. And though I can certainly expect to be paid back, it is essentially a small short-term loan.

Would I be wrong for not passing my savings on to my friends for our annual trip?
```

**Verdict:** KEEP

---

## 110 · `3a2ba569cef19adc` · 1009c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for getting this guy towed?

```text
I'm a recent grad who lives in an apartment with other uni students and I pay to park my car in the garage. The spots are small and there are columns everywhere like most uni garages. description of my spot: To the right side of my spot on the line there is a column, to the left another free spot, then the wall. This dude has parked in my spot a couple times, but I left a note and he moved. I come back and he moved to the spot on the left of me and I know 100% he doesn't pay to park. I know I don't own that spot too, but I picked my spot because I knew no one was going to be to my left and I want to enjoy it for as long as I can. Lemme also note a spot is **$145 a month**. Shit is not cheap, and this guy is driving a new Audi A4. He's also an international student who has a shit ton of foreign government money to go to school here. It's driving me nuts that I'm paying that much to park and I know he hasn't spent a dime to have the same luxury as me.

Would I be wrong for getting this guy towed?
```

**Verdict:** KEEP

---

## 111 · `3ab2428481922170` · 636c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I refused to go to church on Christmas Eve?

```text
My parents stopped trying to get my brothers and I to go to church 2 years ago when we all went off to college,  but this year my mom wants to start going again. She knows that the 3 of us aren’t religious but still wants to go “as a family”. I’m 20, and I really feel like I shouldn’t have to spend 2 hours sitting quietly in church if I don’t want to. This is probably dramatic, but church just seems like a miserable waste of time to me. I was forced to go to church nearly my whole life, and I honestly hated it. I think I’m too old to have Christianity forced on me.

Would I be wrong if I refused to go to church on Christmas Eve?
```

**Verdict:** KEEP

---

## 112 · `3ab6ba1a57b442b4` · 1228c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for calling my brother out on how he treats my family?

```text
For context, im not yet of age to actually pay rent and leave house. Its a single story rancher style house, two bedroom, one bathroom, and one master with a seperate bathroom. We are barely clinging on to the house but we have nine people living there. One person tries to sustain it, my dad, and hes pretty sick of my oldest brother and his partner. They have not paid rent since they've arrived in June of last year and somehow still dont have enough to afford a low income apartment. I've heard no end of complaining from my eldest even though he doesn't pay rent, spends almost all his money on weed, doesn't do dishes, has left his bowl outside three times, and doesn't care for his two children. Since June, my neices are getting heavier treatment and no longer listen to anyone. Many dishes have been broken, our house is messier than ever, and I can never leave my room. I'm scared to talk to him and have not ever responded to him aggresively, no matter how much I want to. This isnt the entire story, but I think its enough to make a decision, all you need to know is this runs seven years deep and ive hated every second being in his presence.

Would I be wrong for calling my brother out on how he treats my family?
```

**Verdict:** KEEP

---

## 113 · `3ae694d46d52d8e0` · 1531c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for not going to a surprise party for a friend if its held on my birthday?

```text
For clarity, I am kind of a people pleaser. I have struggled with self love for a while and I don't know if this would make me a bad person. There's a girl named Sydnie. She hasn't been friends with me for a while for a few reasons. I stood up for an ex she was constantly talking bad about, and she even had people join in and bash him as a group. Anyway, she took upon talking to me (very rarely) and would show basic manners and talk to me about some school related issues. Today she decided to hand me an invitation. She had organized a surprise birthday party for a mutual friend. The mutual friend's birthday was on December 12th. Since they wouldn't be able to celebrate during the week, Sydnie must have decided to do it during the weekend. I thought it sounded kind of fun until I looked at the date and my heart sank. She scheduled it for the 16th, which is my birthday. I felt very malcontent with everything and decided I'd rather not go. I felt like she was basically trying to ignore my birthday and just act like a jerk. I saw her after class, talked to her and told her that I might not be able to go. She said "okay" and then walked away. Do you think that I'm the asshole for deciding not to go? Given, I would go if it were a mutual celebration. TLDR: A girl who treated me badly invited me to a friend's party since it couldn't be celebrated on weekday. I don't feel like going to the party because it's on my birthday

Would I be wrong for not going to a surprise party for a friend if its held on my birthday?
```

**Verdict:** KEEP

---

## 114 · `3b243d757fd3ecc5` · 1940c · ask=title

> **Title** (not part of the prompt): WIBTA: if I’m always the girl who comes between two best friends?

```text
I’ll start by saying I don’t feel like I come between friends in a malicious way or with bad intentions, but I’m always drawn to those situations. 

I’ve never had a solid group of friends, I’ve always kind of drifted between groups. I don’t like drama and I find it gets real old and boring fast. So I’ve always felt better just ignoring it and moving on with my life. No need to get involved and dig myself a grave if I don’t need to. 

Recently I’ve realized I’m always in third wheel situations. Whether it’s with two girl best friends, or girl and guy who are “friends”, I’m always the extra one. 

When it comes to the matter of two girl best friends, I’m always somehow never able to become friends with “the other girl”. I also end up feeling left out most of the time because my friend always seems to be more close with their other friend than me. 

This type of situation happens often with different people. Part of me feels like I’m always that friend who other people talk about all their other friends to, and it’s like everyone has this line they won’t cross with me. Like I’m only allowed so far into friend groups and friendships because people know if I get too far they won’t be able to talk to me the same way about people.. it’s a really weird feeling because sometimes I feel like I’m always the toxic one and that’s why no one wants to really be my friend. 

I don’t feel toxic as in, i talk about others  behind their back. But I’m toxic like I’ll tell someone the truth if I need to. If someone feels comfortable enough with me to ask if their friend talks about them, I’ll tell the truth. 

It seems like in the end the two friends end up against me as if lied to tried to cause an unnecessary situation, but to me it seems both are just lying to each other because neither trust me. 

I’m tired of unhealthy friendships and situations.

Would I be wrong if I’m always the girl who comes between two best friends?
```

**Verdict:** KEEP

---

## 115 · `3b3952c46868de00` · 1104c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I didn’t give my coworker gas money for a 3 hour trip?

```text
Okay so... the title already sounds like I’m TA but if you keep reading it might put some things in perspective. Next week we have a business trip that’s about 3 hours away... cool. My coworker is driving and I said I’d tag along with her. She wants to drive. I can’t remember if she asked me or I asked her but the point is I’ll be in her car to and from this business trip. Since its business, she can get reimbursed for the mileage (a bit more than 50 cents a mile) which is a rate that includes gas. Since our employer is paying for the mileage I feel like I shouldn’t have to give her gas money. Is this wrong on my part? Granted she hasn’t asked but I usually offer gas money for trips even if they don’t, but I don’t feel like it’s necessary in this case. I want this to be known I don’t want to drive but I am perfectly capable of doing so, and there is also company provided transportation to get to the location. But we’d have to Uber everywhere for the two days that we are down there, which would also be reimbursed.

Would I be wrong if I didn’t give my coworker gas money for a 3 hour trip?
```

**Verdict:** KEEP

---

## 116 · `3be4f806523aa066` · 583c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I left a note for my neighbor regarding his trash?

```text
So, I live in a fairly nice/expensive apartment complex in a good area. I get along really well with the neighbor in question 99% of the time. For the last week, he’s been leaving his trash bags out on his porch instead of walking them to the dumpster like a normal person. I know it’s not a huge deal, but it’s a big pet peeve of mine. I feel like maybe I’m overreacting, but SERIOUSLY, I just hate walking by and seeing trash bags. This is a nice complex. Forgive me, but I think that’s pretty.........trashy.

Would I be wrong if I left a note for my neighbor regarding his trash?
```

**Verdict:** KEEP

---

## 117 · `3c5527d38b6edd99` · 1650c · ask=body+title

> **Title** (not part of the prompt): WIBTA for waiting on apology from motherinlaw?

```text
My motherinlaw is in her late eighties. Though we have had disagreements in the past, I always make my very best efforts to mend things and keep open communication. A few weeks back we went to visit during a blizzard, and as I was bringing things in from my car, the garage door stopped working in the open position. I let her know there was a problem and before I could do anything about it, she called her male neighbor over to see if he could fix it. I wanted to observe what he did as he was troubleshooting so I could handle it myself in the future, and my motherinlaw basically told me to go away and that my help wasn’t needed. I explained that I simply wanted to know how to help her in the future and twice she waved me away and said “Goodbye” in a condescending manner. Once the issue was fixed and the neighbor left, she flipped out on me and said that I embarrassed her and insulted her neighbor for asking a question about what he was doing.  It upset me that she treated me this way and I asked why she had to be so hateful towards me. She told me to shut up repeatedly and finally stomped out of the room. I gathered my children and our things and we left. It’s been two weeks and we have not heard a word out of her. Should I apologize for “embarrassing” her, or wait for her to make the first move. I wasn’t trying to be disrespectful, and this is far from the first time she has been rude and dismissive to me. Life is short and I don’t want to deprive the kids of their grandparents, but I also don’t feel like I should be treated rudely over something so inconsequential.

Would I be wrong for waiting on apology from motherinlaw?
```

**Verdict:** KEEP

---

## 118 · `3c9cc47dc08eb4dd` · 1304c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I move out and take my money back from my indecisive uncle?

```text
I am 20 years old and recently moved to where my cousin and her family lives in their home I was originally going to stay with roommates but since I am so close to my cousin, I asked my uncle if I could live with them instead. He originally said no. He and his family were asking for money from our relatives to buy a new house. I offered 2k in my savings to live in the studio that their new house would have. He agreed to this and then said I could live with them until the studio and house is ready. I've been here for three months ad every time he or my aunt or cousin get into an argument (three times now) he tells me I have to go back to my hometown, they're giving me my money back and to pack my stuff up and go. They cancelled their original house bid due to one of the arguments. So now they are looking for a new house and it does not have a studio. (it has a garage that they might convert into a studio but probably not) so it looks like my option is to stay here in this house sharing a room with my cousin. Yes I know I am dumb for offering them money to let me stay here after they said no, but I thought it would be a nice deed for all of us since they needed the money and I do like being with my cousin.

Would I be wrong if I move out and take my money back from my indecisive uncle?
```

**Verdict:** KEEP

---

## 119 · `3cc1fc73aac92593` · 1224c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I called my coworker out for coming in sick so often?

```text
I’ve been at my new job for about 6 months and I love it. The only issue is “Barb”. Barb is very sweet, but she’s *always* getting sick. She seems to have this chronic sinus/upper respiratory issue. I’m not a doctor, but if being a carrier for a respiratory virus is possible... that’s what she is. The common cold version of Typhoid Mary. Every time her “sinus issue” flares up, the entire office gets sick. It’s the nastiest cold imaginable and it just drags on. The last two times I caught it, I was sick for over a month. This is the *third* time that I’ve had the same infection, that hits exactly 2 days after her “sinuses flair up”, since I started. Everyone gets at least 2.5 weeks of PTO, she probably gets more since she’s been here for a long time. She also has the capability to work from home. I have less PTO and my job doesn’t permit me to work from home. So, when she comes in sick, I’m essentially screwed. The cherry on top is that when she’s having these “sinus issues”, she walks around the office coughing and telling everyone how sick she is. At least quarantine yourself in your office! Jesus. This is just getting ridiculous.

Would I be wrong if I called my coworker out for coming in sick so often?
```

**Verdict:** KEEP

---

## 120 · `3d42e9d8d9267117` · 633c · ask=title

> **Title** (not part of the prompt): WIBTA for wanting my parents to pay me back?

```text
recently, i lent my my parents some money because they were tight due to an overdue bill. however, it’s been a couple of weeks since then and my parents owe me a combined $200. they’ve also been paid since then, so i’m not trying to ask them for money they don’t have. this doesn’t mean i’m ungrateful for anything, i just would like some if not all of my money back. i would be more than willing to ask for less from them as well. i feel like an asshole for even thinking about asking, but i currently don’t have a job and won’t for a while, so $200 is a lot of money for me.

Would I be wrong for wanting my parents to pay me back?
```

**Verdict:** KEEP

---

## 121 · `3d54f3c052510205` · 1852c · ask=body+title

> **Title** (not part of the prompt): WIBTA for not hypocritically not sharing test prep materials for the medical school admission test

```text
I'm at a bit of a conundrum. I took the MCAT (medical school admission test) last year and did incredibly well. That isnt a testament to myself but the vast amount of study material and resources I found/made. I have given this to every single friend who is taking it and am happy to help as much as possible. 

However, one person who I am not super close with but have known for years hasn't gotten anything from me. She is really nice to me and has helped in the past with previous classes (I've been told it's due to a crush). She is not like this to others. She is a toxic person who jumps on the opportunity to put down others who want to go into medicine. I know she has withheld helpful materials from people who she doesnt feel would be good doctors. One example is she intentionally didnt tell a girl in a lab she TAd that my lab had an open spot and I'd be willing to meet with her to talk about it. (This girl expressed interest after a lecture one time.) I later learned that this was due to the girl who wanted to be in the lab politely correcting the TA about something, leading to a grudge. 

I dont want to help someone like this. I feel like she would take the spot from someone who genuinely cares about all people at a medical school. At the same time, how can I make that decision? It is a weird spot where I'm not sure what action is the right one. I'm not on an admissions board and I dont know her 100%, so who am I to decide who gets help or not in this process? Now I dont hold the golden ticket, but those who I helped said it was the best resource they had and after talking to her about her plan for the upcoming test, I know it will be helpful. I'm not sure if I would be an asshole for not helping or helping.

Would I be wrong for not hypocritically not sharing test prep materials for the medical school admission test?
```

**Verdict:** KEEP

---

## 122 · `3d7005bdb02c8b5a` · 548c · ask=title

> **Title** (not part of the prompt): WIBTA if I smoked a cigarette in the smoking patio of a bar when there’s a baby here?

```text
So I’m at my favorite bar, in the outdoor smoking patio and a couple brought their baby. They’re sitting about 25 ft away from me. I come here to unwind after work, and that includes smoking a few cigarettes. There’s also and indoor section where they could sit. 

I’m always considerate anywhere I go that’s not a designated smoking section to make sure I’m far away from anyone who might be bothered. I probably won’t do it, but I’m pretty annoyed.

Would I be wrong if I smoked a cigarette in the smoking patio of a bar when there’s a baby here?
```

**Verdict:** KEEP

---

## 123 · `3d9d0ff97ef6e4c2` · 1460c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for telling them to stop making us do their dad's job?

```text
We have some kids that moved into the country to live with their dad. So to help him out, we offered to do their laundry for a short time until he got the hang of things. It's been a few years now and they are still bringing their laundry for us to do. Sometimes they even bring weeks worth of it. A few months ago we told their dad that we could not do it anymore. Now the kids just ask and my mum cannot say no to them so they bring it over anyway. She does not want to tell their dad over and over again. Their dad has been cheating on their mum ever since they got together and introduced his gf to the kids (their mum lives in a different country) and he only takes advantage of us, isn't thankful and once told me that we were not doing him any favors. My mum does not want to cause any issues but this simple thing is causing a rift between us because she just let's it happen even at her own expense (it's a lot of work, expensive etc...) I told the kids once that it is their dad job and they said "ok" but the next time they just brought it over again. I have a feeling their dad is coercing them to do it and I do not think that it is fair since we had told him to stop. Their dad won't answer our calls or respond but we know he gets our messages. We know he has the time because he uses it with his gf's but chooses to pile his responsibilities on us while being ungrateful.

Would I be wrong for telling them to stop making us do their dad's job?
```

**Verdict:** KEEP

---

## 124 · `3de1f6feb3f17b2e` · 1991c · ask=title

> **Title** (not part of the prompt): WIBTA for getting the same tattoo as my friend?

```text
So there’s this song that I really love, and in it there’s a line that I’ve been obsessed with since I first heard it. I told myself that I would get it tattooed one day. I just never had the money, until now.

Well a few months ago I met this girl who I quickly formed a friendship with. She was my first friend in this town since I moved here last new years. We’ve got a lot of shared interests and our personalities are compatible. Overall, I don’t think she’s a bad person or anything, I actually really like her. Anyway, when we were first becoming friends, I showed her the song and told her the tattoo idea because I was excited about it. Apparently that was a bad idea. She loved it. So much so that she got it tattooed herself a few weeks later, before I could.

At the time, I thought, “I guess I’ll just get another line from that song tattooed instead.” I was really upset but she had already gotten the tattoo so there wasn’t much I could do about it after that. 

However I now have the funds to get it tattooed. I decided to say fuck it and get the same lyrics tattooed, because it’s what I really wanted and I don’t want to compromise something important to me and permanent like this. 

I told her, and her reaction was “You’re gonna get that tatted???? Same thing as me??? they have like 500 songs isn’t there a different lyric you could get?” Which pissed me off, because it was my idea. She just had the resources to get it tattooed first. I told her that, and she’s being passive aggressive about it. Obviously, I know she can’t tell me what to do and that I can get those lyrics tattooed on me if I want. It’s not even gonna be in the same spot or in the same font.

Am I being the asshole by still getting the tattoo instead of just letting it go or getting another lyric? This is upsetting me and I don’t want to get the tattoo if every time I look at it I just feel like I was an asshole in this situation.

Would I be wrong for getting the same tattoo as my friend?
```

**Verdict:** KEEP

---

## 125 · `3f04e0d930daed31` · 1390c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I snitched on my teacher

```text
Obligatory on mobile warning, so sorry for spelling and formatting Anyway, my math teacher is quite something, I had him first period on Thursdays and missed the first 20 minutes of a 70 minute lesson because of a music lesson (the recorder btw, this is important  later) and it turns out during my music lessons, he would make jokes at my expense to the whole class, and they were very inappropriate jokes, like jokes about me giving recorders blowjobs and handjobs, as well as sticking them up my ass  The worst part (in my opinion) is that he did this repeatedly, and I only found out the extent of it recently, when it happened over a year ago, and another student had recorder lessons that caused her to miss 30 minutes and there were no jokes about that. Now obviously, jokes being made so publicly at your expense can be very damaging to self esteem and especially if you're not aware of them until after the fact, it was upsetting, to say the least and caused a downward spiral of negative thinking, but on top of all of that, he does the typical teacher shit like singling me out for talking when everyone else is, or when someone behind me is talking, which happens quite often (he doesn't look around he just assumes it's me) and this bastard has the audacity to call ME disrespectful when he yells at me for getting a 98 not a 100.

Would I be wrong if I snitched on my teacher?
```

**Verdict:** KEEP

---

## 126 · `3fa584fb62acd2ae` · 1273c · ask=body+title

> **Title** (not part of the prompt): WIBTA If I told my neighbor to shut the fuck up at night?

```text
I live in a rural area so the houses are about 10-20 meters from each other but every single night for the past years my neighbor starts coughing like crazy with 20-30 minutes intervals, sometimes going for 5 minutes straight.

The cough is horrible to hear, almost sounds like he's about to throw up at any time, even worse is that he does this from 1am to 3am, sometimes even 4 times in a day and I'm not joking when I say that he has been doing it everynight for years.

Some other neighbors say that he uses drugs and smokes everyday but I have never paid attention.

I wanted to record him but I've had no real reason to do so as I don't know what to do with it and I'm afraid of the consequences if I showed it to authorities because many people stupidly think you can't record one's voice without permission in the country I live in.

Sometimes I'd love to just yell at him to shut the fuck up and let me sleep but I'd be waking up everyone in my family by doing so and there's no guarantee he would stop, we also want to be in good terms with neighbors because we want to do some restructuring and we need all of them to agree or something.

I live in Italy if anyone wants to give legal advice.

Would I be wrong if I told my neighbor to shut the fuck up at night?
```

**Verdict:** KEEP

---

## 127 · `3fcb92656e231552` · 1117c · ask=body · scrubbed

> **Title** (not part of the prompt): WIBTA Hostage Vacation

```text
We have spent the past few days doing what my dad, my brother, and my mom wants. For some more back ground, I asked a few days ago if we could visit a famous chocolate place a few blocks from our hotel, and we went, but not until 5:30 (the store closes at 6). Afterwards I told my mom thanks for taking me but could we go again another day when they aren’t about to close, but she said no. Then we spent the whole next day and today doing activities for my brother and dad again. Tonight I asked if we could could quickly walk a few blocks and see Jenna marbles wax figure because I look up to her a lot, but I was told that everyone else was tired and we needed to go back to the hotel. We are only here for a few more days and I’d like to do something I’m interested in but all my ideas get shot down. After keeping quiet for so long I want to ask my family why no one wants to do anything I’m interested in but I don’t want to be rude if I’m in the wrong. Should I just suck it up and appreciate how I spend my spring break or is it okay if I ask to do an activity for me? I feel like a hostage in my own vacation.
```

**Verdict:** KEEP

---

## 128 · `402baca2e48c05a3` · 399c · ask=closer

> **Title** (not part of the prompt): WIBTA Making Friends for a purpose

```text
I want to get better at certain languages (spanish, korean and few others), I thought of getting up with some people because the using speak those languages. I wanted to befriend them but I was told that would be using them. I believe I am fine because I do want to be friends but with the added benefit of learning languages from born speakers which is way better than online stuff

Help me decide.
```

**Verdict:** KEEP

---

## 129 · `40a489e04393c579` · 536c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I sold a gift from a friend?

```text
To start off, the obligatory mobile user set up stuff. Ok so I have not done this, but was considering it. Around a month or two ago me and my friend were very into CounterStrike Global Offense (CSGO for short). In CSGO, you can open cases and get skins for your items. The rarest skins you can get are for your knife. He had 3 knives (he bought them) and he decided to give me on. Now, we don’t play CSGO as much as we used to and I was thinking of selling it to get other steam games.

Would I be wrong if I sold a gift from a friend?
```

**Verdict:** KEEP

---

## 130 · `40fd453b7f10e4d3` · 1595c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for sharing advice with my cubicle neighbor regarding a sensitive problem I overheard?

```text
My office is a cubicle farm, and my cubicle is on the edge of my work group's zone. My neighbor who is not my coworker is "Joanne", and she's a very nice woman as far as I can tell. Like many cubicle farms, the unspoken rule seems to be that you ignore anything you hear that isn't directed at you. However, you still can hear...and I've heard Joanne talking to people on the phone (presumably family members, based on context) about a problem her college-aged son is going through. Very specifically, he has tried to drop a semester for medical reasons, was denied, appealed, and then was denied again. The medical drop is under particular circumstances which aren't important to the question. Now, four years ago I got a retroactive medical drop from the same university in the same circumstances, and based on the details of what I can overhear, this person is going about the documentation and appeal process *all wrong*, and based on my experience I can give some advice on how to restructure it and some additional types of documentation to pursue that would drastically improve the chances of a future appeal being accepted. This coworker seems nice, and I hate to have someone with a failed semester on their record just because they don't know how to document. On the other hand, the circumstances are potentially very embarrassing to my coworker, and I don't want to make her uncomfortable or upset. Is there a way I can pass her advice that would minimize the level of assholery?

Would I be wrong for sharing advice with my cubicle neighbor regarding a sensitive problem I overheard?
```

**Verdict:** KEEP

---

## 131 · `41dff8aee535e4ba` · 852c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I refused to be friends with someone because they have acne?

```text
Okay so I met a girl at uni and she’s so lovely and we get on pretty well. But she has really bad acne. Like, super inflamed blackhead clusters ALL over her face. I have trypophobia (fear of clusters/holes) and her face sets me off really badly. I get anxious and itchy whenever I see her and I can’t look at her properly when I talk to her without panicking. I feel so bad about it, especially since I have (not as bad) acne myself and I know what it’s like to be insecure about your face. 

We’ve been messaging and she keeps trying to meet up. I’ve been dodging it so far and I feel awful. I think I’m going to have to politely tell her I can’t be friends with her but I’m not sure how yet. My friends are acting like I’m a massive asshole for this - are they right??

Would I be wrong if I refused to be friends with someone because they have acne?
```

**Verdict:** KEEP

---

## 132 · `42131f5482f56e9d` · 618c · ask=title

> **Title** (not part of the prompt): WIBTA if I don't message my coworker to see if she's ok?

```text
My coworker hasn't rocked up to work today. She's never ever late, in fact, she's usually here an hour early. 

Whenever she has been off sick, my boss calls to tell me she's not coming in. When she has an appointment, she tells me or puts it in our shared calendar.

I really don't like her and, quite frankly, I don't want her to come in but I don't want to look like an asshole for not checking up on her if something is actually wrong. 

We're a tiny company (only 3 or 4 of us including her who work from the office) if that makes a difference.

Would I be wrong if I don't message my coworker to see if she's ok?
```

**Verdict:** KEEP

---

## 133 · `422cd35e560d0cae` · 956c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told if I told my friend he was being a self absorbed jerk the day after his grandma died (for unrelated reasons)?

```text
Me and my friend are both 14. Yesterday, he told me when he was angry that him and some of my friends prefer to hang out without me. I obviously got upset, and told him as such. He never really apologized, and today at school made fun of me for being upset and for some particular things I said (I didn’t really feel like eating lunch with him the next day and things of the such). He was just an overall jerk about it. He is also a pretty narcissistic person. However, I don’t know if I should tell him as such as his grandmother died yesterday (although he isn’t to upset about it, if that matters). TL;DR- my friend said something upsetting to me yesterday, and made fun of me for getting upset and for what I said. I want to tell him off, but his grandma died yesterday, although he doesn’t appear to be too upset.

Would I be wrong if I told if I told my friend he was being a self absorbed jerk the day after his grandma died (for unrelated reasons)?
```

**Verdict:** KEEP

---

## 134 · `43778d1a02a6e630` · 588c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I reported a group of women for violating quiet hours?

```text
So I’m a university student who lives in dorms and my room is right next to the lounge (unfortunately). Anyways there’s this group of five women who all come together at around 10:30ish and stay there until 1-2 am (quiet hours start at 11). They’re extremely loud and obnoxious and I’ve had to have asked them to quiet down at least 10 different times but they continue to be loud and obnoxious every night, it’s gotten to the point that if I end up going to bed at 10:00 pm I’ll be woken up by them at midnight.

Would I be wrong if I reported a group of women for violating quiet hours?
```

**Verdict:** KEEP

---

## 135 · `43b11883cebd96ec` · 607c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I don’t play with my friends and ruin everything?

```text
I’m a kid age 15. That’s all the background that’s necessary. So I went to my friends house along with my brother. We are really close and have been with each other since basically birth. However, today when I came over I really had no energy. I only play indoors and never go outside unless I’m told too. We went downstairs and booted up The PS4, but I wasn’t feeling into it. So I quit after a while of playing. They also got off and when I say that I didn’t want to play anymore they yelled at me saying I ruin everything every time.

Would I be wrong if I don’t play with my friends and ruin everything?
```

**Verdict:** KEEP

---

## 136 · `43c9ab7d6cc9b357` · 1985c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I blared my alarm really loud in the morning to make my flatmate as sleep deprived as me?

```text
I have this flatmate - let's call him Jim. I live with Jim and two others in a shared apartment. Jim and I have two separate bedrooms next to each other on one side of the apartment, while the other two have their own bedrooms on the other side. So it's only me who hears anything from Jim's room at night. Jim has the tendency to stay up until 3 to 4 AM playing video games with his friends very loudly. I've asked him to at least keep quiet after 2 AM two times already, but both times he's lasted less than a month before starting back up again. He's complained that since his gaming friends are mostly European and he usually has classes in the afternoon he can only play with them in the middle of the night,. He's also made passive aggressive comments in the group chat about how "the Internet is so much better in the middle of the night." and has lashed at out me for doing simple things like asking him to clean his dirty dishes after they've sat in the sink for a week. I'm getting the sense that confronting him directly a third time will just start something and I would rather not live in fear of retaliation when our doors don't have locks. I'm already leaving the apartment in less than two months, but finals season is next month and I don't want to be sleep deprived during that. I've tried earplugs, but I can still hear him through them. I've tried listening to music, but I can't get to sleep. Sometimes I manage to sleep, but then I'm jolted awake because my earplugs or earphones fall out during the night and I hear one of his screeches. Since I have 8 AM classes every day and have to wake up anyways, I was thinking of leaving my alarm blaring at full volume for the 15-30 minutes it takes for me to get ready, figuring if he's forced awake as early as I am, he won't stay up so late. I know it's kinda a dick move, but I'm just so tired. Literally.

Would I be wrong if I blared my alarm really loud in the morning to make my flatmate as sleep deprived as me?
```

**Verdict:** KEEP

---

## 137 · `43ea9fa48e43d3d9` · 803c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for asking my friends to pay me to play my board games?

```text
Title doesn’t explain the situation fully. Me and a group of friends get together every so often and play board games together of which I am one of the sole providers (of games that is). I feel I have poured hundreds of dollars into these games but they get mistreated by some of the players which I can understand to a degree. Would asking my friends for 15 dollars to play a new “legacy” game I purchased make me an asshole? Legacy board games have to be played with the same people who are committed and after the campaign is over it cannot be replayed the same way. Would just like to recoup some of the money spent on the game to reinvest into a future legacy game for the group. But I’m getting a bit of backlash from some.

Would I be wrong for asking my friends to pay me to play my board games?
```

**Verdict:** KEEP

---

## 138 · `44d134167552e545` · 1109c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I anonymously told my new neighbors not to smoke weed where I can smell it?

```text
I've lived in my townhouse for a couple years now, and the nice old guy in the adjacent unit moved out and was just replaced with neighbors I haven't met yet. Tonight, I was sitting in my living room and started smelling weed. Curious, I opened my back door a crack to investigate and got hit in the face with a strong smell of weed coming from the neighboring unit. I don't smoke. I've never smoked. I have no interest in smoking. And I don't want to smell smoke in my house. Marijuana is not legal where I am, but I don't want to narc on my neighbors. I would just prefer that if they're going to smoke they do it somewhere that I can't smell it. Functionally, this would mean they'd have to do it inside their house or garage, or somewhere else away from home, since I'll smell it if they're on their deck or in the driveway. I don't want to get into an altercation, so I'd probably just leave a note at the community mailbox. The next point of escalation (which I'd rather not do) would be involving the HOA.

Would I be wrong if I anonymously told my new neighbors not to smoke weed where I can smell it?
```

**Verdict:** KEEP

---

## 139 · `461c0a83eddfe03c` · 1563c · ask=body+title

> **Title** (not part of the prompt): WIBTA if i didnt return an elderly lady's 10 dollars?

```text
So this happened about 5 or 6 years ago. My mother, sister and I always went to this park that was close to the close to the school we went to. We always went to the park after school and my mom would talk to her friend and my sister and I would just go on the playground.

One day as I was playing tag with some other kids at the park, I spotted a 10 dollar bill near a tree. I rush over to pick up the bill and stuff it in my pocket. Now, my family at the time was poor, not super poor but below average in income so I didn't want to return it to the principal or give it to my mom.

I kept playing with the other kids and I notice an elderly lady was looking near the area I found the dollar bill. I casually walk towards where my mom who was distracted talking to her friend the entire time I took the money and sat down next to her. I just watched as the lady tried to find what I assumed was the 10 dollars. 

I definitely would've returned the 10 dollars if the lady had asked my mom or I as she was looking close to where my mom was sitting.

The lady looked for about 4 or 5 minutes and eventually left. I felt bad after the whole situation and told my mom what happened. She told me I didn't do the right thing and should've returned it when I was still at the park. My older sister called me a douchebag for not returning the money.

Would I be the AH in this situation?

Tl;dr: Found 10 dollars at a park, saw a lady trying to find something (most likely the dollar), didn't return it.

Would I be wrong if i didnt return an elderly lady's 10 dollars?
```

**Verdict:** KEEP

---

## 140 · `4647e5b9b0b352c5` · 795c · ask=body+title · scrubbed

> **Title** (not part of the prompt): Wibta if i hosted bbq and was anti-social?

```text
They show no love towards her, they barely invite her to their functions and yet she's expected to host here and there and puts out one hell of a spread. They show up. They drink all the booze they take inventory assess the situation. They asked a lot of really personal questions and then they lie if not to be heard from again until they want to come over and mooch some more. So I tend to grill for drinks relax keep quiet and I leave them alone. I'm not awkward around them just more like anti-social. It's a quarterly Endeavor that results in fake conversation. They don't really care about us and have never shown otherwise. They're second cousins and there kids and spouses and such. So does this make me the a-hole for being silent?

Would I be wrong if i hosted bbq and was anti-social?
```

**Verdict:** KEEP

---

## 141 · `46bf9b851d575afd` · 1244c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I told my brother and sister-in-law who just had a baby to pull their heads out of their asses?

```text
My brother and sister-in-law just recently had a baby. She was born on January 24th and they are very great parents. My only problem is that neither one of them have a job. The "job" they had was a bullshit excuse for a company a friend of a friend made that made them COOs but didn't pay them for 6 MONTHS. They still have no money and are relaying on family or friends. They are just lazy in my opinion. They both left their jobs to go to this "company" and then never questioned why they weren't getting paid. They both almost refuse to get a job because of the baby actually being here now an they have had to get money from people to buy them food and pay their bills because of this. On top of this they want to take legal action against the guy for them not getting paid. I know that the whole family has been on their asses and have been telling them how dumb they are being so I have been quiet and just let them come to me if they need to rant or whatnot without me preaching at them. I'm not sure what I would say yet (I would keep my cool and not blow up) but I don't want them to feel like no one is on their side.

Would I be wrong if I told my brother and sister-in-law who just had a baby to pull their heads out of their asses?
```

**Verdict:** KEEP

---

## 142 · `4704ab46656efeb3` · 863c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I accepted casino cashier paying me extra $25 after warning her she gave me too much?

```text
Title pretty much says it all. Played some poker, was cashing out for $343 after a 1/2 session. Lady behind the counter counts it out to $368 (She must have thought I had 5 green $25 chips). I look down at the chips I gave her and do a quick count (there's 40 $5 red chips on a little rack I handed her, 4 green $25 chips, and another 8 $5 chips + 3 $1 chips equaling $343. I say "m'am I don't want you to get in trouble but you're giving me too much money, it should be $343". She seems a little distracted, counts it again and comes up with $368 again. I say "m'am, there's 200 in the rack, 100 in green chips next to it, and 5...10...15...20..25..30..35..40..41.42.43 outside the rack. So she finally recounts it and thanks me, and I go on my merry way.

Would I be wrong if I accepted casino cashier paying me extra $25 after warning her she gave me too much?
```

**Verdict:** KEEP

---

## 143 · `470a8ca1b8f05375` · 651c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I didn't invite my parents to my wedding?

```text
Over the last year some very important events in my life have occurred, First one being the first in my family to graduate college. And since then two more big things have happened (not relevant) and one to come in a few weeks which they haven't declined (yet). They live only two hours away so its entirely feasible for them to take a Saturday afternoon to drive to see this even for their son. Its gotten me so disheartened that if I were to find a woman of marriage quality i'm tempted to not invite them just to send the message that they have hurt me, not to cut them out of my life.

Would I be wrong if I didn't invite my parents to my wedding?
```

**Verdict:** KEEP

---

## 144 · `47362267a50b4d45` · 1002c · ask=title

> **Title** (not part of the prompt): WIBTA if I ask my neighbor to stop singing?

```text
I'm in college and recently moved into my first apartment all by myself. The building is an old house that was converted into apartments, and I was the only one in the building at first. Fair to say that I really appreciated the quiet for a bit. 

I did not realize until someone else moved into the apartment next to mine just how thin the walls are in this building. I normally wouldn't mind, I get that the walls being thin are not her fault and I hope that she will be ok if I have people over to my place some night, but her singing is just really getting to me. 

It's on and off almost every night (including weeknights) from around 6 till 10pm, and I am an engineering student with morning classes and finals week coming up. While it does bug me, I get that it's her home too and that she at least normally stops before 10.

I'll also go ahead and add that it is not bad singing, it's very pleasant actually, but it is distractingly loud.

Would I be wrong if I ask my neighbor to stop singing?
```

**Verdict:** KEEP

---

## 145 · `47a6d85bb5c235ac` · 2059c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I don't let a stranger sleep in my room

```text
Hey everyone! I answer a lot of questions here so it's nice to finally ask if what I am thinking of doing is the right thing or not. I live in a nice apartment in the downtown area of a major city. I rent this apartment (as in I'm not the owner just a tenant) and three other people also live there (none of them are the landlord). It's a good setup for me as I'm a college student, the room has it's own bathroom, the rent is a bargain, and the guys I live with are pretty cool. Right now I don't have classes so I came back to my hometown where I am busy studying for finals. I recently got a phone call from a housemate. He told me a friend of his is coming to stay with him for a few days. That is not the problem. It's perfectly normal at our house for this to happen. We have this foldable bed that can be wheeled into our rooms for guests or they just stay in the living room in that bed (what usually happens is this). However, my housemate has asked me if I the guy can crash in my bedroom. And I have a problem with this. I don't know the guy (though he doesn't sound like a troublemaker), it's my room (which I guess sounds petty and probably is but it is my place) and it's not really clean (it's not dirty as I usually clean it weekly but the last week at college was crazy with all the group projects and tests so I didn't have time to clean). Besides, the bed is fitted with my sheets and all. It is worth noting that my housemate says he will make my bed with sheets that belong to him and will clean the room. But I'm weirded out by the thought of him going in there and clean the place, the bathroom and change the bed sheets. I don't really want him to do that. And I don't want some guy I don't know sleeping in my bed. It's weird. Why can't he sleep in my housemates bed or in the guest bed at the living room? It's the way it always is. I don't get why this guy should get special treatment. I would never ask anyone for this. So, I'm supposed to call him with a reply today.

Would I be wrong if I don't let a stranger sleep in my room?
```

**Verdict:** KEEP

---

## 146 · `47cea74330be6443` · 1579c · ask=title

> **Title** (not part of the prompt): WIBTA by asking a friend for alcohol after an awkward situation?

```text
So there’s this person that I recently met at a college party. We both hit it off, and while we don’t text often or meet up, we’re usually pretty friendly in passing, and they’ve offered to buy me alcohol im the future since they’re 21. We ended up at another party together, and things got awkward. Since she had been drinking at other events, she let me and a friend have most of her drink. It was late into the party, so I was already kind of drunk.

After a while I was hammered. My friend came up with the bright idea of playing matchmaker, and asked me if I’d want to make out with her. Being very I drunk I agreed, and they went about setting us up. Needless to say, it failed, and the entire situation was very awkward, although I didn’t try to force anything physically (I never would).

The next day I texted them as normal but after a little they left me on read. I felt so guilty and horrible, but I didn’t know how they felt, and I didn’t want to come out with some huge apology that they would see as even stranger. I haven’t really seen this person at all since that party, but there’s another coming up soon that we would both maybe be at. I feel like it would be horrible to ask her to pick up something for me (I would be paying her obviously) after what happened, but I hate mooching off other friends, and I’m not great at parties where everyone is drunk and I’m stone cold sober (which this one would probably be based on the people going). How dickish would this be? I feel so bad

Would I be wrong by asking a friend for alcohol after an awkward situation?
```

**Verdict:** KEEP

---

## 147 · `48ac2f7eeae6d8b2` · 981c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I didn’t contest a parking ticket?

```text
So recently on a Sunday night I got a parking ticket for being parked in a space that became a no parking space after midnight. Thing is, I parked right next to a sign that clearly stated “no parking 8-5 Wednesday for street cleaning” and the sign about no parking past midnight was at least 100 feet away (I didn’t notice it when parking). I was going to contest the ticket with photographs showing how I clearly wouldn’t be able to see that sign, but the officer got my plate wrong. I only realized this because I tried using my plate number to contest the ticket. The ticket has my car described completely incorrectly (different year, plate, and last 4 VIN). There’s no way that the ticket can be tracked back to me afaik UNLESS I contest it. The way I see it, the officer made mistake on top of another mistake and it seems like I have to deal with that. If I choose not to contest it, I may damage his standing at work.

Would I be wrong if I didn’t contest a parking ticket?
```

**Verdict:** KEEP

---

## 148 · `491b116237b652f2` · 2162c · ask=title

> **Title** (not part of the prompt): WIBTA for trying to get a family with their autistic son and visibly fake service dog banned from the theatre?

```text
to start, I go to musicals and plays very often. my family has season tickets that we buy each year so we're able to see all the shows that come. the tickets for this season is in the same row every time. 

behind us, there is a family with a severely autistic child (not child, he's in his 20s) who makes noises the entire time, kicks everyone's seat, screams, and tries to sing along with the songs by going "BLEHBLEHBLEH". as well as this, the parents talk throughout the entire play to their son and what's happening, very, very loudly. even if we ask them to stop, they won't. it's disturbing everyone that sits remotely near them because they're so loud. someone from across the theatre last night actually shushed them very loudly. not only does he try and sing along, he also makes very loud farting noises with his mouth. 

now onto the "service dog". I like dogs. I'd be fine with watching the play with the dog behind me, the dog is very cute. but their service dog is very fake. she growls at everyone walking by, will not listen to commands (i.e. yesterday the mom told the dog to sit, she didn't, the mom had to push the dogs butt on the floor), has a visbily fake service dog vest ordered from eBay, and is disruptive. the dog is also seperated from the autistic kid (we asked questions about the dog, the dog is the autistic kids), which is NEVER supposed to happen. the handeler and the dog should never be seperated (I'm getting a service dog soon, so I know this). we asked about the tasks and the lady froze up. she got really anxious and just said "she alerts me when something is going to happen" (the point of a service dog is to provide independence for the handler, so this contradicts) then quickly shut off the conversation. there's a lot more with the dog, but I'll stop there. 

I feel bad because the autistic guy clearly enjoys coming to these, but the entire family is super disruptive to everyone at the theatre. Hamilton is the play after next, and I don't want to deal with them at Hamilton especially :(

Would I be wrong for trying to get a family with their autistic son and visibly fake service dog banned from the theatre?
```

**Verdict:** KEEP

---

## 149 · `4a299b760e944346` · 1620c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if i told my roommate to stop using my stuff

```text
I live with one other person that I’ve lived with for the past two years. She and I lived in the same dorm for a year, not roommates, and then in the same apartment for two years in separate rooms. Before this year, we discussed getting our own dishes but decided to go in communally on things like pots, pans and large utensils such as spatulas and ladles. However, I ended up bringing everything. But I don’t mind her using my pots, pans, etc., because those are agreed upon as shared. However, although she brought her own plates, bowls and utensils, I frequently come home to see her using mine, which are nicer, even though we agreed to bring and use our own. I didn’t say anything, because they’re just dishes and she always washes them. But the other day, I came home and saw her eating using my condiments. Now, if she used, say, a little bit of ketchup, I wouldn’t mind, but these are soy sauce and sesame oil I went specifically to the Asian market to get. She never asked if I could use them, and then didn’t put the soy sauce back into the refrigerator after she used it. Also, some of my snacks went missing from the pantry a week ago, and I was puzzled, and then I saw it reappear earlier tonight on her shelf. I feel like these are such tiny things that aren’t worth mentioning, and I would gladly let her use it if she asked, but since I’ve let her go months without saying anything I feel like it’s rude to bring it up now. TLDR; roommate has been using utensils that we didn’t agree to share & has been using my condiments without asking.

Would I be wrong if i told my roommate to stop using my stuff?
```

**Verdict:** KEEP

---

## 150 · `4bab63ddc33e534a` · 614c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I didn’t give away my Wii to a poor single mother with 2 kids

```text
I’m on mobile so sorry for an poor formatting A bit of background: I’ve had my Wii for about 10 years or so. I still occasionally play games on it but the sentimental value this thing has to me is utter insane. I was basically grown up on this thing. Recently my mom asked me if I would be willing to give it away to single mother at her work. She has two twins and their birthday is approaching and she wanted to get them a gift. I argued that you could buy a Wii for pretty cheap and I couldn’t imagine games cost more than $10.

Would I be wrong if I didn’t give away my Wii to a poor single mother with 2 kids?
```

**Verdict:** KEEP

---

## 151 · `4be2b071a20e3719` · 1049c · ask=body · scrubbed

> **Title** (not part of the prompt): WIBTA with the use of my vehicle?

```text
My friend and two of my friend's friends are going camping. They usually rent an RV but can't afford one this year so they're driving and bringing tents. They have invited me. I'm totally down to go, but I really want to rent an RV (which I can afford, but it would be a splurge) because there's no way I'll be able to comfortably sleep in a tent and I hate sharing public showers. I have a bad back and difficulty adjusting to unfamiliar sounds at night so a white noise machine in an RV is a must. I also can't fall asleep in the same room as anyone I'm not close to, and I'm not close to this group but they are truly lovely people. They are lovely people but they are slobs about their living environment and I don't want them making the RV gross (tracking mud/sand/whatever in etc) or costing me a deposit. Should I just skip this trip? I'm imagining possibly being hit up for refrigerator space, showers, and sleeping spots (especially if it's a bigger/nicer RV and/or if the weather goes south) and I'm just super uncomfortable with the idea.
```

**Verdict:** KEEP

---

## 152 · `4c473dac241b5eb5` · 971c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I didn't let my downstairs neighbor in after he got locked out

```text
I live in an apartment that is basically a house divided into 3 apartments, with only one main entrance that leads to the apartments. One of the guys who lives downstairs always forgets his keys when he leaves, like 3-4 times a week I get a call to come let him in. This isn't usually a big deal as I'm home a lot of the time, although sometimes I have had to leave where I was at to unlock the door, and sometimes he calls late at night. I also suspect that he is purposefully leaving the main door unlocked when he leaves, because many times when I leave that door is unlocked, and I always make sure to lock that door. Today I almost just ignored his call to teach him a lesson, but then I gave in about 5 minutes later. When I got outside he wasn't there waiting for me, no, he was trying to break the window into his apartment so he could get in(not the first time he's done this).

Would I be wrong if I didn't let my downstairs neighbor in after he got locked out?
```

**Verdict:** KEEP

---

## 153 · `4d3207855a613c16` · 472c · ask=body+title

> **Title** (not part of the prompt): WIBTA for offering my secret santa money to go over the $ limit?

```text
So my group of friends does secret santa and the cap is $20. We often play board games together and we have kind of exhausted the $15 range of board games (Resistance, Coup, One Night Ultimate Werewolf, Codenames, Spyfall, Expansions for all of those).  

I included Deception: Shadow of Hong Kong in my wishlist which is $30ish and offered to chip in $15 if they choose that item. Is this rude?

Would I be wrong for offering my secret santa money to go over the $ limit?
```

**Verdict:** KEEP

---

## 154 · `4dec312a919f136c` · 668c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I bought an electronic noisy toy for a 2nd birthday party gift?

```text
I'm thinking age appropriate toys such as alphabet/color singing things like a piano keyboard that helps learning. 

This is my  bestfriend's first kid and it's her babies 2nd birthday. I don't have kids so maybe I'm a little out of touch. But my intention is to get things the kid can play and learn at the same time. 

My friend has jokingly said she wouldn't allow it from anyone but maybe I'd get a best friend auntie pass for one. 

She's having a big party and I find it hard to believe that nobody will get electronic toys.

Is it practical for my friend to try to avoid them?

Would I be wrong if I bought an electronic noisy toy for a 2nd birthday party gift?
```

**Verdict:** KEEP

---

## 155 · `4e135dea47c79701` · 513c · ask=body+title

> **Title** (not part of the prompt): WIBTA for asking someone who chooses the bathroom stall next to mine to choose another stall?

```text
I am a very shy pooper, and if someone can hear/smell what I'm doing in there I get very self-conscious. I realize this is my burden to bear, and no one should be forced to comply, however, in a theoretical scenario, if there are 7 stalls, I am already occupying one, the rest are empty, and the next person chooses the one directly next to mine, would I be an asshole to ask them kindly to move upon entry?

Would I be wrong for asking someone who chooses the bathroom stall next to mine to choose another stall?
```

**Verdict:** KEEP

---

## 156 · `4e1a0e1e1f113290` · 554c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I don't go to dinner with my aunt tomorrow?

```text
My dad met his bio family, and while I'm super happy for him, I went to dinner with them tonight for the first time. 

It was awful. I was stuck on the other end of the table and totally ignored by everyone. I tried to engage but would just get one word answers from the aunt.

His bio mom is pretty chill but I legitimately cried at how awful it was tonight.

Should I suck it up and go tomorrow or say no? They flew down from Florida and she'll be meeting my brother and cousin at lunch.

Would I be wrong if I don't go to dinner with my aunt tomorrow?
```

**Verdict:** KEEP

---

## 157 · `4e310ceb460ad734` · 492c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I didnt tell my brothers its my birthday

```text
Every year for our birthdays my brothers and I have been going out to eat for each others birthdays. I don't live with my family for reasons. Anyways I call my brother and tell them what time do they want to go out. Then they tell me they don't feel like going out today. I always remember theirs and remember to tell them each and every year. I really do love my siblings but it hurts when they forgot they have another sibling.

Would I be wrong if I didnt tell my brothers its my birthday?
```

**Verdict:** KEEP

---

## 158 · `4e727ee3ce64ffd0` · 1037c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I got a PhD student in trouble for not paying cover at a bar?

```text
I work a club on the weekends as a bouncer. We have a $9 cover on Saturday nights, and while it's high, a lot of people still pay it. I was working our side entrance (which has 1 bouncer and 1 register clerk) when a group of college students got in line. I checked IDs for 3 of them, and continued down the line. 2 women (1 of which I checked in), stopped for a second at the register, and then proceeded to take advantage that we were really busy, and decided not to pay. I saw this and ran past them, and they also started running. They tried losing me in the club, but I caught back up to them, and told them they needed to pay. One girl said "yeaaaa... sure we do", and then sprinted off again. I lost track of them, but got one of their names from our system. While she's banned from entering the club, her name is very unique, and it didn't take but a quick google search to find that she's a PhD student researching at a nearby university. Thanks!

Would I be wrong if I got a PhD student in trouble for not paying cover at a bar?
```

**Verdict:** KEEP

---

## 159 · `4e8a70be99eb856e` · 374c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for reporting my teacher?

```text
My ethics teacher decided to put on the “documentary” ‘Climate Hustle’ in class, if you don’t know the documentary it’s basically about climate change and how it’s a hoax. My problem is that a teacher should never share let alone push his political opinion onto impressionable students, especially an opinion as dangerous as this.

Would I be wrong for reporting my teacher?
```

**Verdict:** KEEP

---

## 160 · `4f0d677da1b29443` · 632c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA For asking a girl to prom that I know my friend wants?

```text
We both want to ask the same girl, I've wanted to for a few months, but neither of us knew the other wanted to until yesterday. When we talked about it, we both kind of just brushed it off like "rip, I should've told you" and "what? I thought you liked someone else" then I had to leave. Thing is, he's my friend, but he's not a super close friend. I had already told all my close friends I wanted to ask this girl over a month ago. So I don't know where we stand. I'm really not trying to hurt my friend's feelings but I've wanted to ask this girl for months

Would I be wrong for asking a girl to prom that I know my friend wants?
```

**Verdict:** KEEP

---

## 161 · `50ed2fe663f792a7` · 1341c · ask=title

> **Title** (not part of the prompt): WIBTA if I commented on a customers order so they would tip for once?

```text
Reading the title now it kind of sounds really bad LOL.  

TL;DR customer is on the 3rd floor, no elevator available, always places large orders and never tips. I want to make a comment next time I deliver so he would start tipping.

My work delivers for multiple restaurants kind of like Postmates / Uber eats. Us drivers make wages by delivery fee commissions and tips alone. It's really not a lot, min wage (8 some) where I am is low compared to cost of living, this is my second job. I make $5 out of 7.99 with the minimum delivery fee. We have a time frame of 10-15 to deliver but most of the time it's always sooner. If something is wrong with an order we are more than willing to go back to the pickup place. We do not touch your order by any means. This one customer lives close by to most of the restaurants we do so his fee is the minimum every time. He always places big orders and is on the 3rd floor with no elevator access. All customers are aware they can tip when placing an order online and such it's an option before you check out. I really want to know if I'd be an asshole if I made a comment next time I deliver just so he could think about tipping. BTW his order equivalent fills up my largest bag which is more than half the size of me!

Would I be wrong if I commented on a customers order so they would tip for once?
```

**Verdict:** KEEP

---

## 162 · `518148cd60fe783c` · 965c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA For not filling my friends name on an Important School task

```text
Have to fill this in quick because i'll have to make it within 2 Hours and Send it to the Teacher. Me and My friend both being pretty lazy decided to make this together. We postponed it multiple times and today is the deadline. I wanted to make the Task together on school multiple times, but he either had to "go somewhere" or "forgot" to stay on school. He knew we had to make it today and he called in sick. I went home and told him to ask for another postponement. He said okay and guess what. He forgot about it. I told him he could help me since we still have 3 hours. He said "I'm about to go to bed. I still have Migrain." I didn't buy this since i myself have had Migrain since i was little and i know it doesn't last all day. I told him i'll make it myself but tell the teacher i made it myself so he won't get a grade. Causing big problems for him. But its his own fault imo.

Would I be wrong for not filling my friends name on an Important School task?
```

**Verdict:** KEEP

---

## 163 · `518cba4017429f62` · 939c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I didn't want to pay for a hotel room that I won't be staying in?

```text
One of my best friends is turning 21 in the coming weeks, and our group of about 7 guys (all 21+) had the idea to rent 2 hotel rooms in downtown for the night, drink, go to a strip club, and maybe hire an escort for the birthday boy for the night. Split between 7 of us for **just the hotel rooms**, the cost would be around 75-100 per person. This doesn't cover alcohol costs, escort costs, or the $25 in one dollar bills we were going to give the birthday boy to splurge at the club. I would assume the night in whole would come out to around $200-$225 pp. Normally I would have no issue paying this. However, I have an early work shift the following morning and am not planning on drinking, or staying the night in the hotel. My plan is to leave the hotel around midnight - 1 A.M. so that I can have at least a couple hours of sleep before my shift.

Would I be wrong if I didn't want to pay for a hotel room that I won't be staying in?
```

**Verdict:** KEEP

---

## 164 · `51f66c98b149427f` · 2036c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA to tell my dad to eat healthier or refuse to let him eat candy?

```text
My father has eaten his whole life too much which is the reason why he has always been overweight. My grandmother didn't really know what eating normally was due to the fact that she has experienced food scarcity in WWII. She didn't want her children (my father, aunt and uncle) to live in the same conditions as she used to, so she let them eat as much as they could throughout their whole childhood. About ten years ago, my dad's doctor confirmed that he has type 2 diabetes which, at that time, seemed to deeply affect his mood. For example, he became upset when a family member told him he couldn't eat ice cream. Many family members confronted him and told him he should eat healthier, by which he starts to play the victim and saying things like "you don't know what it's like to have diabetes". His reaction confused me, because he already knew his health was in bad shape and he acted like it was some sort of joke at first. In the past, he followed harmful diets that made him lose and gain weight quickly, like juice cleanses. He says it's difficult to lose weight and I don't doubt that, but to me and my family it seems like he's barely trying to improve his lifestyle. He follows a diet that his doctor recommended to him due to his health condition, but eating custard and so-called healthy cookies after dinner? That doesn't sound like something a doctor would recommend to someone with diabetes. He walks an hour a day, yet he eats junk food at work (not everyday, fortunately). My mother can tell this by his expenses that she can search on his bank account. I don't know exactly how much he eats, but I can tell he doesn't know what eating normally is. My family and I are at our wit's end on how to improve his health. I'm worried he might get a heart attack one day and I don't know what I could say or do to make him eat normally, because he's quite stubborn about this topic. Whenever he asks for candy and such, I want to say 'no'.

Would I be wrong to tell my dad to eat healthier or refuse to let him eat candy?
```

**Verdict:** KEEP

---

## 165 · `5215bd266f1c012b` · 448c · ask=body+title

> **Title** (not part of the prompt): WIBTA for giving a kid better gifts than their parents?

```text
Let's say that hypothetically I make a considerable amount more money than my siblings and have no kids of my own to spoil. Would I be an asshole for giving my nieces and nephews gifts that are significantly better than what their parents can give them?

Are there any roles this would be majorly different. What if it was a friend's SO who I'm also good friends with for example?

Would I be wrong for giving a kid better gifts than their parents?
```

**Verdict:** KEEP

---

## 166 · `5295b6e7d58c433e` · 944c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for reporting a neighbor that refuses to leash their dog, saying that it acts aggressive on a leash?

```text
To clarify, I've never run into the dog or owner, but someone posted on the Nextdoor app about leashing your dogs, as it's become a big problem in my neighborhood. One person continually states that they are "willing to take the risk" of their large dog being off leash, stating that it's easier for them because the dog is fearfully aggressive when on a leash. I find this attitude upsetting, as he's even stated he's okay accepting the risk of the dog running in the road. We live in a large city with a lot of traffic, which does have a mandatory leash law. I also don't trust that a dog with aggression issues would be safe off the leash all the time to everyone else. I have his name and general location because of the posts, and screenshots of the comments made about the dogs behavior and the "risks" he has accepted.

Would I be wrong for reporting a neighbor that refuses to leash their dog, saying that it acts aggressive on a leash?
```

**Verdict:** KEEP

---

## 167 · `531932064740496f` · 1771c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I made a point to tell my roommate he got me sick right before my trip?

```text
Basically the situation is that I have a huge trip that I'm leaving early next week, and people know how big of a deal this is to me. People also know I'm a bit paranoid about getting sick, but for good reason because I seem to catch colds really easily.

Early this week my roommate started coughing and sneezing a lot, and being really haphazard about it just sneezing everywhere, so I kept asking him if he thinks he's getting sick because if he does think so that I'll go stay with my parents so I definitely don't get sick. He kept saying no no it's just allergies yadda yadda, but then two days ago it was completely clear he was sick, so I asked him "So you're clearly sick" and he said yes and that he didn't want to worry me. I didn't make a big deal, even though I was very pissed, I just left a couple hours later and now I'm at my parents.

Wouldn't you know it, now I'm sick.

We're pretty close friends/roommates, so I feel like I should tell him... Something. That I don't want to make a big deal about it but he got me sick and it's really frustrating, and in the future just be up front with me if he's sick. And stop using the paper towels in the kitchen to blow his nose. If I don't tell him I'm sick, he might never realise he fucked me over because I'm staying at my parents this week... Because I'm sick.

I don't know if this is creating unnecessary drama/makes me an asshole, but it's a big deal to me and I'm very upset he lied to me and got me sick when I would have been a lot more careful if he had been up front.

For the record, it's definitely him who got me sick because I've barely left the house this week, I've just been scrambling to prepare.

Would I be wrong if I made a point to tell my roommate he got me sick right before my trip?
```

**Verdict:** KEEP

---

## 168 · `5328d2e1595d7536` · 650c · ask=closer

> **Title** (not part of the prompt): WIBTA? I want to ask my mom not to bring her bf to my place.

```text
So, I recently moved into my own place from my mom's. She asked to come visit, and that's okay, but I'd like to ask her not to bring her boyfriend. He's a nice guy and all, but she acts like a totally different person when she's around him... meaner actually (to me). It's the 'I don't need you now that I have a bf mentality'. And I definitely don't want to stop her from living her life, but there's unnecessary negativity coming from this situation.

That's not really an energy I want to bring into my new home and I don't want to have to look at her being a totally different person in my place.

I'm not sure what to do here. What do you think?
```

**Verdict:** KEEP

---

## 169 · `5340feeb34cd56a7` · 1001c · ask=body+title

> **Title** (not part of the prompt): WIBTA for not helping my family cleaning up my grandparents basement?

```text
So, to make thing short my mother recently made some comments about how when i get home the next time we could ,as a family ( my mother, my stepfather, my sister, her husband and my uncle and me), clean up the complete basement of my grandparents. She said it would !only! take a week to do. ( large house 30 years of woodworking and hoarding of \*soon\* reconstructed furniture my grandfather is hoarding and buying to resell) Im probably only there for 2 weeks.

now for the problem: Im currently working in a different country only coming home every 4-5 months at best. the reason why i can visit my home so frequently is because my employer pays for the trip back. The reason why he pays for it is because i dont work mon-fri. i work every day of the week. So my "vacation" is in reality my weekends i worked for.

I would like to use that time to meet up with friends and relaxing.

would that make me the asshole?

Would I be wrong for not helping my family cleaning up my grandparents basement?
```

**Verdict:** KEEP

---

## 170 · `5375b6aee4ca9914` · 1196c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I didn't tell my boss I'm being considered for a promotion on a different department?

```text
I've been working on this company for a year now and my position is literally a bullshit job, I have a minimum amount of responsibilities and I'm treated like a child (I'm the youngest in the department). I've asked several times for more things to do, told my boss (let's call him Pete) to give me a chance to prove myself and he only tells me to "be patient", my coworkers say they've been there and that I should wait, but I feel stuck here, I've even been looking for other jobs for a while. Someone from a different department told her boss (let's call her Sarah) that I have potential and they should transfer me, so they have given me small tasks and I've tried my best to excel.

Recently the Sarah told me she wanted me to work with her, but to don't tell Pete about it yet, that "someone" will talk to him. I told everything to one of my coworkers and she said I'm going behind Pete's back and it's unprofessional and unethical. I don't want to get on Sarah's bad side by doing what she explicitly told me not to, but I don't want to disrespect my current boss. Am I the asshole?

Would I be wrong if I didn't tell my boss I'm being considered for a promotion on a different department?
```

**Verdict:** KEEP

---

## 171 · `53f0b5a2da1d0762` · 626c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for refusing to pay my entire share of our heating bill?

```text
There’s more to the title than that. I’ve never been late on bills or rent and have, on more than one occasion, covered for my roommates. I’ve been out of state on an internship for 2 and a half weeks, and somehow our heating bill is astronomical. I live with 3 other roommates (4 total in the house) and we always split everything evenly. Except now that I’m away I don’t feel it’s my responsibility to pay my entire portion. I believe I should only have to pay half of my portion (my share would be $50, but I believe I showed only have to pay $25).

Would I be wrong for refusing to pay my entire share of our heating bill?
```

**Verdict:** KEEP

---

## 172 · `544c9762904aa440` · 446c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I didn't show up for work tomorrow?

```text
I'm scheduled to work tomorrow morning. I'm a security guard at a mall in a really good part of town. Good coverage on a normal day is three people and we will have double that the next few weeks for no immediately discernible reason. This for shift is also an overtime shift, so I'd be getting paid time and a half. I've thought about not showing up just to see if they would even notice.

Would I be wrong if I didn't show up for work tomorrow?
```

**Verdict:** KEEP

---

## 173 · `557cef7c0c978731` · 1425c · ask=closer · scrubbed

> **Title** (not part of the prompt): WIBTA not realizing I might have gotten an internship I recommended someone else for

```text
I'm a junior in University studying Mechanical Engineering. Engineering students live and die by their internships, trying to get a full time position without one can be terribly difficult. I have prepared and have been fortunate enough to accept an internship for summer 2019. During my time as a student I came into contact with a smaller company's CEO. After I explained my desire to land an internship he tried to help me by giving me notices of various openings at his company. For a variety of reasons, things never aligned and I never worked for his company. Our university career fair was this past week and my CEO friend was there. I said hello and told him of my success of landing an internship with another company for the summer. He was happy for me and said, "too bad, because we have a problem that a mechanical engineer could help us with." He asked for a recommendation and I gave him the resumee of one of my friends. After speaking with said friend, it is a part time position that they are looking to fill right now. Maybe I misunderstood and thought it was a position for the coming summer. Maybe I'm just an idiot for not grabbing the opportunity while it was in front of me. It has occurred to me that maybe they wanted someone to work from now until the end of summer or any other time. Would it be worth rolling the dice to try to get this part time position until my next internship

Help me decide.
```

**Verdict:** KEEP

---

## 174 · `55c9cc5dc40b0b49` · 1553c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I asked a friend to compensate me for the bonus he caused me to lose?

```text
About 2 months ago a a friend of mine was working for Doordash and I was working for Postmates. He convinced me to switch to Doordash because of a promotion. After I completed 150 deliveries within the 60 day limit, he would receive a $1500 bonus and I would receive a $100 bonus. We agreed that I would be getting kinda fucked over, especially because Postmates is a much better platform. (better pay, more flexible, much less technical and logistical issues). We agreed to split both bonuses so each of us would receive $800. The other day we were texting and the bonus came up, conversation went like this.

Friend: You think you'll get it by next week?

Me: Yup.

Friend: Only one prob, I blocked the account.

Me: What does that mean?

Friend: I don't know if I'm gonna receive the promo.

Me: It bothers me that you're just telling me this now, like I still feel you should give me the $800 if you don't get the money.

Friend: Yeah... no, you're on crack.

To be clear "I blocked the account" meant he blocked the number that sent him notifications and deleted the app. I don't think this is at all gonna be a problem, I'm mainly posting this cause I'm interested in the hypothetical. I fell he would owe me as I have been working for both of us the last month and a half. Without the bonus I would be losing money considering I would be making more with postmates, also I have dealt with a considerable amount of bullshit with Doordash's faulty platform.

Would I be wrong if I asked a friend to compensate me for the bonus he caused me to lose?
```

**Verdict:** KEEP

---

## 175 · `56374ea879b132b9` · 580c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I didnt invite a friend of a friend to a party

```text
So me and my group of friends are all fairly avid stoners and I'm having a party for 4.20 this year I'm inviting a few people round, about 6 or so. My best friend currently is very good friends with a person I haven't respected or liked for a long time, let's call this person jack and my best friend joseph. I don't like jack for multiple reasons but the main one being that he has previously stolen from my house and from another of my friends  and lied about for months even after we confronted him about it.

Would I be wrong if I didnt invite a friend of a friend to a party?
```

**Verdict:** KEEP

---

## 176 · `57e1392e992f8967` · 1048c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for not letting my brother use my car to learn how to drive

```text
Hello, I'm a 20 year old student from Norway. I recently (as in November last year) bought my first "real" car with my own money, an Audi A4 2010 with \~167 000 km (\~104 000  miles). I bought a manual car because it was a good deal on a nice looking used car, I'm used to manuals from my previous beat up car and because I believe manual cars last longer. This is where my problem begins, my brother is just turning 16 this year and at that age is when you start student driving here in Norway and you get the classic L sticker for your car. My mother nor my father has a manual car of their own and they think i should let him learn how to drive a clutch in my new (used) car. I do not like the prospect of letting him do this  because I don't want him to ruin the clutch or something similar. Since the car already has a good mileage i feel like him using for a couple years learning to drive clutch will wear it out fast and that's something I don't want to happen.

Would I be wrong for not letting my brother use my car to learn how to drive?
```

**Verdict:** KEEP

---

## 177 · `580ef179924b112a` · 1415c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I didn't invite my friend to a vacation?

```text
Some buddies and one of their families and I went on a spring break trip recently and one of my other friends did not get invited and was kind of bummed out. It was not a "you cannot go" reason because it came down to space as we were driving about eight hours. Friend A (whom invited us and parents were driving), Friend B and Friend C and I all got to talking about what to do next spring break and we're thinking about a cruise. I asked if we should invite anyone else and Friend A immediately knew who I was asking in reference to. A said, "I'd be fine with it but you guys are together a lot, would you be willing to?" and I realized it was true. With some friends at college, we (myself and friend D who wasn't invited) were together a lot including school and the gym. He really doesn't go out on his own and refuses to hang out with others unless I'm going or I invite him. Looking back now I sometimes invited him to get him out because he wouldn't otherwise, but I don't want to force my other friends to have to put up with him on a vacation just because I'm afraid of his reaction if I weren't to invite him. I also wouldn't know how to tell him. Yes, he is friends with the other group, but not as close as I am to them. We also are all ok with keeping the group as small as possible to keep things simple as more people complicates things.

Would I be wrong if I didn't invite my friend to a vacation?
```

**Verdict:** KEEP

---

## 178 · `581d2c0839ba0700` · 578c · ask=title

> **Title** (not part of the prompt): WIBTA if I spilled the beans about my cousin's marriage?

```text
My cousin and his girlfriend eloped over last weekend and are planning on telling my parents on Christmas with the whole family around. I am 90% certain my hothead father won't take this news well for various reasons and I want to tell him about it before then to avoid any extreme reactions.

I understand it is not my place to release details about my cousin's life, but this our family is going through an unprecedented amount of drama this Christmas in particular and I want to do what I can to combat it.

Would I be wrong if I spilled the beans about my cousin's marriage?
```

**Verdict:** KEEP

---

## 179 · `582913d2a74ba66e` · 874c · ask=body+title

> **Title** (not part of the prompt): WIBTA for thinking this?

```text
So this all started when the school year started. It is currently the 3rd quarter for me.

My friend and I have been friends for a long time.  Like maybe 7 or 8 years. But now we seem to be drifting away. And I don't want to tell her to spend more time with me because that would just be rude. We are in this group of friends where she really knows everyone else very well while I don't know them so well. She spends lots of time with them, but I'm usually left behind. What I'm trying to get at is that she has started spending more time with other people but I don't know if I'm being jealous or kind of overprotective. So what should I tell her, should I just keep staying laid back and keeping to myself, or should I try and be more social/tell her. 

Point of story: Am I the asshole for thinking this or is she forgetting about me?

Would I be wrong for thinking this?
```

**Verdict:** KEEP

---

## 180 · `5926500177d29b6d` · 1043c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I go through with my plans of going overseas after I’ve started seeing a girl?

```text
So a couple of months ago I met a girl at a club and went home with her. At the time I thought it was just going to be a one night stand, but we’ve stayed in touch. We message each other a few times a week. She’s stayed at my house a couple of times and we spent New Years together. It’s very casual, We’re not in a relationship, and we haven’t agreed to be exclusive or anything like that (although we both have been exclusive since meeting, because i think we’re both just like that)  The thing is, since before I met her I’ve been planing on going overseas for a couple of months at the end of January. I’ve already quit my job in anticipation of this. I can tell she really likes me, and I feel like she’s going to be a little hurt if I leave. We’ve talked about traveling, as it’s a common interest, and I have brought it up, but I think it sort of came across more as something I want to do (rather than something I was 100% going to do)

Would I be wrong if I go through with my plans of going overseas after I’ve started seeing a girl?
```

**Verdict:** KEEP

---

## 181 · `594bcfe35759051f` · 1811c · ask=title

> **Title** (not part of the prompt): WIBTA for asking landlord to help with cost of food lost due to faulty appliance?

```text
Hi guys! I will have been at my apartment for 2 years in May. For reference, it is a duplex owned by one dude, not a chain or a big business. There are lots of little issues with the place, but I mostly put up with them because there is really no other place for me to live in the area.

The issue: Back in June 2018, the fridge that had been here when I moved in failed. I was gone for a few days before discovering this which meant I had a lot of hot, rotten food to clean out of the fridge and freezer. After I cleaned it, I called the landlord and asked him to either fix or replace the unit. He came by a few days later and did so, with a fridge that looked even older/busted up than the last. But I figured whatever, as long as it does the job. It sucked to be out of \~$100 of food but I was just grateful to have the old, now stinky fridge out of my kitchen.

Until today, when the replacement fridge broke. Completely stopped cooling/freezing, and I tried trouble shooting it, making sure it was plugged in, the doors were shut etc. But nope, it was just broken. I feel like he just keeps replacing faulty appliances with faulty appliances (repairs/appliances are his responsibility as per the lease). Now I have lost about $200 worth of brand new groceries, and have rent coming up next week. I cannot afford to replace any of these groceries and pay rent. Would it be unheard of/completely make me an asshole to ask to help for food lost because of his bad appliances/have it come out of this month's rent? I did not even consider it the first time, but now I feel like I should ask because this seems to be a pattern and I'm worried that I'm just going to have more ruined groceries in 6 months-year. TIA!

Would I be wrong for asking landlord to help with cost of food lost due to faulty appliance?
```

**Verdict:** KEEP

---

## 182 · `595e21035d84d2fd` · 949c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I leave a note on my neighbor's car asking them to park better?

```text
I live in a small apartment complex with assigned parking spaces in a carport in the alley behind our building. The alley is narrow, and I struggle to back in to my spot (I have an end spot with the carport wall on one side and a couple of metal columns on the other, and because there's a Dumpster in the alley, I can only get into my spot from one direction). I'm not a very good driver anyway, but the spot is definitely challenging. I don't know who is assigned the spot next to me. Usually there is only a small car parked there, but several times I've seen a larger SUV that also backs into their space. The problem is that the SUV is never backed in completely--they usually have at least a good three feet to go. This means that the massive front end of the SUV is sticking out into the alley, making it really, really hard for me to back around their car.

Would I be wrong if I leave a note on my neighbor's car asking them to park better?
```

**Verdict:** KEEP

---

## 183 · `5a981c65278e5b7a` · 973c · ask=title

> **Title** (not part of the prompt): WIBTA if I told my roommate her boyfriend couldn’t stay over?

```text
So basically my relationship with my roommate isn’t great. We don’t have much in common and she’s messy, inconsiderate and kind of annoying to me. But despite this stuff I try to be a decent person and not do anything rude to her and just get through the rest of this semester (we’re both in college and I’m moving out at the end of the semester). 
Anyway, the reason I don’t want her boyfriend staying over is because she invited him without telling me he was coming after I’ve asked her to let me know. Also, she spent nearly the entire day cooking for him in our (very small) shared kitchen so I wasn’t able to use it for hours. She also let him stay at our apartment all day watching Netflix. 
I’m planning on basically saying that she can’t invite him over if this is how they’re gonna behave. Also I’d like to know if there is anything I can do legally to get him out if she invites him again.

Would I be wrong if I told my roommate her boyfriend couldn’t stay over?
```

**Verdict:** KEEP

---

## 184 · `5ad821a71a3d0bc9` · 607c · ask=title

> **Title** (not part of the prompt): WIBTA if I bring my own beer without sharing?

```text
This one is mild, but I still want to know if I would be rude if I did this.

So, I’m going over to my friends house tonight, and usually I buy a 12 pack of cheap light beer for everyone, but I never get drunk off of it. Recently, I’ve been getting into pricier beer of quality, some of which actually get me drunk! Would I be an asshole if I brought that fancier beer to my friends house just for myself AS WELL AS a 12 pack of cheap stuff for everyone else? The reason I wouldn’t be sharing is that it’s A) expensive and B) a single 1.5 pt bottle.

Would I be wrong if I bring my own beer without sharing?
```

**Verdict:** KEEP

---

## 185 · `5b602e4973bec584` · 466c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I sold my buddy weed at a slight profit?

```text
My buddy wants to buy a half oz of weed off me. I have a really good connect which means that I can get his half oz for alot lower than market value. Would I be an asshole for charging him 10$ more than I paid for it? It would still be an extremely good price, and tbh if I was to sell it I could for 20-30 dollars more than what I'd be charging him, but would that be kind of a dick move if he's my boy?

Would I be wrong if I sold my buddy weed at a slight profit?
```

**Verdict:** KEEP

---

## 186 · `5b9f9633c269ec9c` · 1669c · ask=title

> **Title** (not part of the prompt): WIBTA for quitting my job?

```text
Okay, the reason I’m posting this is because I have mixed results from my coworkers and friends. I’m on mobile, so sorry for any issues. TLDR @ bottom

Some backstory; I’m a 16 y/o shift manager at a fast food chain. Over the summer I was working 50-60 hours a week, and when school started I said I can only work Friday, Saturday, and Sunday. This was obviously a blow to the management team, which consists of two full time managers (GM and AGM) and then another part time student who only works Sunday . 

Here is where the controversy comes in. I want to quit very badly. The reason is that I asked off 2 weeks ago for this coming Friday because I have finals this week (my school runs trimesters, weird right?) and I want to work on papers that are due at midnight. So my manager, when doing the schedule last thursday, got mad at me and told a coworker that she’s going to schedule me for Wednesday because i shouldn’t have asked off for a day that I usually work. 

Well, I’m screwed now because I have finals and a presentation due on Thursday and I will be working to ~11 Wednesday night, when I specifically said at the beginning of the school year I couldn’t work schooldays. My coworkers are saying I’d be an douche because I would leave the management stranded with no closer for Friday thru Sunday, especially seeing that we are starting to get busy now that the weather is picking up. 

TLDR: GM got mad at me because I asked off for a day I usually work because I have finals, so she schedules me a close in the middle of finals week.  Our management team would be fucked if I left, but I want to quit. 

So WIABTA

Would I be wrong for quitting my job?
```

**Verdict:** KEEP

---

## 187 · `5ba59697fc01ed2b` · 790c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I sold furniture that was given to me?

```text
So, I wasn't actually involved in this apart from being a bystander. But basically, my parents gave my brother a couch when he moved out, and he sold it.  Or rather...his girlfriend sold it. They moved out together, so I suppose it was given to both of them. I don't know if it makes a difference. They didn't have it too long before they sold it, and it wasn't like it was old or worn out. 

Anyways, it was a few years ago, but I still get to hear the complaining once in a blue moon. Was it a dick move on their part? Or were they being overdramatic that they sold it? I mean, I assume it wasn't being lent to them so I don't know if they expected it back, or what. Who the hell *lends* a couch anyways? What do you guys think?

Would I be wrong if I sold furniture that was given to me?
```

**Verdict:** KEEP

---

## 188 · `5c0974f71d331bea` · 916c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked this guy why he ghosted me?

```text
So I meet this guy. We hit it off really well, he seems really into me. We talk all day every day for over a month and things are getting intense. He tells me he isn’t seeing anyone else and doesn’t want to be seeing anyone else. Literally does a 180 after that. Pretty much he stops replying and seems to completely lose interest. I was blindsided. Now when he sees me around he ignores me. I don’t know what I did wrong and it’s eating me up inside. Should I ask him why we don’t talk anymore, or should I just take the message as a loud and clear demand for me to fuck off? I seriously don’t know what I did and I think I would get some closure if I asked him, but I don’t want to come off as harassing him or whatever. I am just amazed that he lost interest literally overnight and I can’t believe it, I want to know if there’s another reason behind this.

Would I be wrong if I asked this guy why he ghosted me?
```

**Verdict:** KEEP

---

## 189 · `5cd59c552fd31a4e` · 849c · ask=title

> **Title** (not part of the prompt): WIBTA for calling my sister out on her massive dumps?

```text
Apologizing in advance for the subject matter. 

My sister and I share a bathroom. For the most part, we have no issues. The only problem is she ALWAYS clogs the toilet, I’m talking two or three times a week. If she took care of it, I wouldn’t even be posting this, but she never unclogs it herself. In the past, I’ve politely asked if she could fix it when it happens, but she can never even own up to it, saying it wasn’t her. We are the only ones that use that bathroom. I tried bringing it up to my mom, but she enables the behavior and just says she’ll take care of it herself. I want to skip the politeness and demand she act like a normal human being and clean up after herself like a functional adult, but I’m worried I’ll come out of that situation looking like the asshole.

Would I be wrong for calling my sister out on her massive dumps?
```

**Verdict:** KEEP

---

## 190 · `5cf5f83e8de1625e` · 604c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I didn’t let my neighbor use my grill?

```text
Some context: I am/was a furloughed employee and my next door neighbor asked to use my grill to cook hot dogs for furloughed workers. He didn’t know my grill is charcoal and it’s a pain in the ass to start because the top vent is cast iron which I’ve greased up. It’s currently below freezing although it’s supposed to get in the mid 40s. When it’s cold it’s extremely difficult to open the top vent and if you don’t open the top vent the charcoal goes out. To compound this, he hasn’t checked to see if we were okay at all during the shutdown.

Would I be wrong if I didn’t let my neighbor use my grill?
```

**Verdict:** KEEP

---

## 191 · `5d367a0bba0858cb` · 1120c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I started asking for gas money?

```text
Background info: My brother (21M) and my friend (24F) don't have cars and don't drive. My brother is working on getting his license. My friend has had her learner's permit a couple times but she's working through some other stuff so it's not her top priority. My brother's at university, the same one my friend and I both graduated from. I (23F) have a car, so whenever I do something with either of them I'm the one who drives (obviously). The three of us are taking a trip together, about 6 hours of driving total. My friend left some stuff with her aunt that she wants to pick up, so she asked if I could pick it up for her next time I'm there. I'm dropping my brother back at college after his spring break and I asked if she wanted to come with us. I don't want to spring an unexpected expense on them. I've never asked them for gas money before and they've never offered. Neither of them has a job and I'm working full time, which is why I'm not sure if I WBTA. But it's not a trip I would make by myself and I feel like they should be contributing in some way.

Would I be wrong if I started asking for gas money?
```

**Verdict:** KEEP

---

## 192 · `5d7b1eaa04092a05` · 509c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I stretched out and napped

```text
I was on a flight the other day and it was underbooked, so only the window and the aisle seats were filled, but the middle seats were all empty. I wanted to take a nap and the middle seat looked so inviting, but I didn't want to disturb the other person in my row by taking up two seats. I'm short so I wouldn't have gotten into her seat space at all. Would it have been rude to lie down while I slept? Or was it for the best that I dozed in a seated position?

Would I be wrong if I stretched out and napped?
```

**Verdict:** KEEP

---

## 193 · `5d99bdf7d4193e55` · 440c · ask=body+title

> **Title** (not part of the prompt): WIBTA If I asked to return a gift?

```text
My friends got me a coat for Christmas that I don't really like. They all chipped in some money and gave me the gift and I really do appreciate the gesture, but the present isn't really my style and so I don't think I would wear it.

I've seen in TV and films that people sometimes they ask to return or exchange presents but I've never seen anyone do it in real life. Would it be really rude?

Would I be wrong if I asked to return a gift?
```

**Verdict:** KEEP

---

## 194 · `5db14eb4c25554ff` · 1448c · ask=body+title

> **Title** (not part of the prompt): WIBTA for interfering in professor's lecture when he starts telling offensive jokes?

```text
One of our college professors, among others, likes to joke when lecturing. Unfortunately, some of his jokes are quite disrespectful (i.e. "Ultrasound is emitted by women who get their whims denied") - and, unlike other joke types, these also don't bring anything useful to the lecture. I've tried telling him about it several times but he dismissed my concerns with indifferent "Uh-huh, sure, of course it's not funny" and, clearly, it did no impact. 

I can't skip his course or just be absent because, while not really important, it's still mandatory and he requires us to show notes of his lectures for a grade.  
I don't think bringing this up to faculty leaders will result in any effect because telling such jokes isn't a crime or something, so why would they bother.  
I can, technically, suck it up with "ah, this guy's just a prick" - but while I want to have a good grade like any student, I would still also like it very much if professors didn't belittle others, including myself, in the process of teaching, no matter how petty their insults are.

So, I want to get more vocal, maybe even mirror the joke as last resort, in trying to signal him the message or at least get it off my chest (nevermind the *quod licet Iovi, non licet bovi*) when he says something like that again. Would that, however, make me no better than him in the end?

Would I be wrong for interfering in professor's lecture when he starts telling offensive jokes?
```

**Verdict:** KEEP

---

## 195 · `5dc6aa64843f30ed` · 401c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I ask my brother to start paying for this house?

```text
Last year I signed a house over to my brother. We had an agreement that he would pay me 200 a month for a year or so for it. At the time he was having some money troubles, so I told him we'd decide later when he could start paying. We haven't really talked about it since then, but I know he's in a better situation now money wise.

Would I be wrong if I ask my brother to start paying for this house?
```

**Verdict:** KEEP

---

## 196 · `5df4ab76c25f4009` · 912c · ask=closer · scrubbed

> **Title** (not part of the prompt): WIBTA - Neighbor’s young child very loud

```text
Sidebar: I’ve actually taken to calling the child Thor God of Thunder because I can hear the kid’s thundering footsteps at all times of the day and night and I don’t actually know the kid’s name. Thus far, I’ve let it go without saying anything to the neighbors or to the building management (who can handle noise complaints like this). I figured the kid is a toddler and you can’t expect them to be quiet right? Well, it’s getting distressing. It’s 2AM right now and the cat and I were just woken up for the fifth time this week to very loud kid footsteps, sounds associated with falling down, and the kid scream crying. I also work from home where I have to work with video and audio files, so the noise is not conducive to my work. The loud noises seem to be particularly distressing to my timid rescue cat - who went from purring in my lap to hiding under the bed at this latest incident.

What would you do?
```

**Verdict:** KEEP

---

## 197 · `5e058d25d9b5d4a2` · 1555c · ask=closer

> **Title** (not part of the prompt): WIBTA quitting a job I just started three weeks ago?

```text
So let me start this off by saying I’m NOT a lazy person, I grew up in a low income family and have always worked as hard as I can at any job I find myself doing. 

That being said. 

Three weeks ago I started a job in a dog daycare as a dog handler. I got this job because the owner of the business is a family friend and she knew I’d been looking for a second job for quite awhile. When I started the job I received barely any training, despite the fact I’d never worked with dogs in a pack setting (they knew this and were perfectly fine with it, they said they would give me all the training I need). 
I was told I’d only be working 15 hours a week, this was a dealbreaker for me since it’s my second job and I need time for my other job- this past week I worked 28 hours. Apart from that the job is very very stressful to the point where during some shifts I find myself going into the bathroom and bust out crying. On a normal day we can get around 50 dogs- it’s just me in a room with them and with that number of dogs theres constant fighting (which I was never trained on how to break up a fight). 
I feel so disappointed in myself that I’m considering quitting so quickly. I’m afraid that quitting will strain my families relationship with the owner. I also don’t want to quit because I know it won’t look good on a resume, and my other job is in a field that is overlooked. 
Any and all feedback would be appreciated!

Tl;dr: I can’t stand my new job but I’m afraid of quitting over something that isn’t worth quitting over.

What would you do?
```

**Verdict:** KEEP

---

## 198 · `5f27159eadc80cbe` · 1446c · ask=body+title

> **Title** (not part of the prompt): WIBTA for telling my friend to not stay with her day?

```text
Ok so I I'm more asking if my friend wbta if she decided to tell her dad she doesnt want to live with him.

She is 17 and lives with 50/50, her dad and step mom half of the time, and her bio mom the other half. She loves her bio mom and her dad, but hates her step mom. She is constantly rude to her and tells her, her problems aren't real. For example she was born without a hip and had sergery as a baby. But as a result she has tremendous pain most of the time, but her step mom tells her it's not real and to get over it. She has pain meds but step mom gets really mad if she takes them. 

Her mom also gets really mad at her for the most ridiculous things, like getting fingerprints on the fridge. I have seen it first hand and I was blown away at how the step mom reacted. She yells at her if she finds a pine needle in the house. Just in general really small stuff that is insane to be punished for. 

She has tried telling her dad about it but he doesnt seem to care and when she talks to her bio mom she just says she should figure it out with her dad. So it seems like she is stuck. But it's gone on for so long she despises going to her dads. I think since she is 17 she should have more say in where she stays especially if she is miserable for half of her life. 

So would she be the ass whole for wanting to go to her moms house full time or atleast more than 50/50?

Would I be wrong for telling my friend to not stay with her day?
```

**Verdict:** KEEP

---

## 199 · `6009d755586b1102` · 507c · ask=body+title

> **Title** (not part of the prompt): WIBTA for getting a tattoo that’s similar to one that my sister wants?

```text
I’m studying abroad and a local tattoo artist posted a flash sheet of flower tattoos. I’ve always thought they were pretty and was thinking of getting one to commemorate my time here. When I told my sister, she said it wasn’t cool because she wanted something like that as well. This isn’t news to me, I knew she also wanted one, but I don’t see how my getting one detracts from her eventually getting one. Am I the asshole?

Would I be wrong for getting a tattoo that’s similar to one that my sister wants?
```

**Verdict:** KEEP

---

## 200 · `60de4be5e3a4eb14` · 1126c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I told my close friend she has literally no chance of getting into med school?

```text
So I've known this girl for about 3 years now. We met during college orientation and have been close friends since then. She was premed from the start whereas I'm non-traditional and started taking premed courses after freshman year. We've only had two courses together and in both instances, I performed exceptionally better than her, despite my tutoring her and giving her my study guide (I'm talking about A- Vs D+).

Earlier today, she brought up median MCAT/GPAs for a lot of colleges and she talked about how she even though she wasn't close, she knew she would get into one of them. I really want to tell her to strongly reconsider as...well she doesn't really have a chance. Her GPA is close to a 2- 2.5 (based on a resume I saw a few months ago) and she has literally no volunteer/clinical work but she says she'll change it in the summer...

I really want her to wake up and smell the roses and think about something else because she really doesn't have any chance of getting in. Should I tell her how I really feel?

Would I be wrong if I told my close friend she has literally no chance of getting into med school?
```

**Verdict:** KEEP

---

## 201 · `61552f58181c763c` · 1364c · ask=title

> **Title** (not part of the prompt): WIBTA for complaining about stinky coworker?

```text
For some background, I’m essentially a glorified 1:1 in-home caregiver, with 2 clients to the home in this case. We generally do our own thing, but it’s a shared space.

My coworker for one shift literally smells like, idk a months-old sweaty piss bottle, *all the time*. Their step-mother is a peer in the company as well, and I’ve overheard her trying to talk to them about it on the phone a few times, but it’s still 100% always an issue. Like, it’s not just that they smell so bad you don’t want to be around them, they smell so bad the whole room STANKS for around 10-15 minutes after they’ve left it. It’s almost gag-inducing, and I regularly clean up actual human waste without issue. Even my client will occasionally refuse to enter common rooms for a good 10 minutes after they’ve been in there. I understand people have their own health and hygiene stuff going on sometimes (and I don’t always smell like roses myself) but man, I’ve honestly never encountered such a... flagrantly fragrant... person before (and I’ve been to smash bros tournaments).

I don’t want to be an asshole and I’m pretty uncomfortable with even bringing it up but seriously, I don’t know what else to do at this point. I feel like someone needs to say something, I can’t imagine their client exactly appreciates it either.

Would I be wrong for complaining about stinky coworker?
```

**Verdict:** KEEP

---

## 202 · `61d007600ab99a6f` · 1819c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I didn’t listen to a complaint I got at work?

```text
I work at a place that doubles as a grocery store and pizza/sandwich place. On the weekends I prep the food for when we open later in the night. Since our store is so small we have to go next door to the university kitchen and use the equipment over there. It’s a nightmare I have to live every week for about 4 hours. It’s always crowded and loud. But what really upsets me is that there is this one guy who always takes up space (2-3 out of the 5 tables, several pots, spots on the cooling racks, etc) including mine, to get his work done. I’ve found his food in my containers several times, my pans/food/utensils moved (even on the floor), and my stoves turned off, for example. This weekend in particular I had a lot of stuff to prep so I used more dishes than usual. However, I was also cooking faster than usual and spaced out my work so that I wasn’t using a million spaces at once. I also usually take up a quarter or half of a table that’s in “his” space, but this time I was someplace different. I would say I was in his space for a cumulative time of 5 minutes in order to use a large pot. Tonight my supervisor takes me aside and says that the guy complained to her,saying that I was taking up too much space and that I needed to stop. She admits that another supervisor present said that I wasn’t doing anything wrong, but apparently it was enough for someone to go behind his back to come next door, instead of asking me directly for more space (at least). I’m not really willing to slow down and cook one thing at a time to accommodate just one person (other people would ask if I was finished with one thing, and I would ask them, everything respectful) but I don’t really want to cause drama (he seems to be well known in the kitchen).

Would I be wrong if I didn’t listen to a complaint I got at work?
```

**Verdict:** KEEP

---

## 203 · `623a0a774c3d2967` · 2056c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for blowing up on my roommate's friend?

```text
The first time I ever met this guy, the second he walked through the door he mentioned a quilt I had folded over my couch in the front room was ugly. "Wow, that's an ugly blanket!" I wrote it off as maybe he was just socially awkward and was looking for something to say and that's what came out of his mouth cause he was nervous(He was early 20's at the time). The quilt was a gift, made from my deceased Grandmother, who loved to sew/knit/quilt in her spare time. I didn't press the issue and just let it roll off, my roommate mentioned it was something my Grandmother made, and there was no apology; I later mentioned to my roommate that it was pretty rude of him to say after he had left. Another instance when he came over he mentioned again about the ugliness of the blanket, I bit my tongue and just rolled my eyes. There were a few more people this time and he brought his hookah pipe. I didn't partake in any cause I don't smoke tobacco, but he went to the kitchen to get it all setup. The kitchen is kind of my space with my roommate situation. I do all the grocery/cooking so I always maintain the kitchen and it's cleanliness. After they had left, there was a huge mess of tobacco all over the counters, on the floors, and some smudge marks on the drawers that I had to clean. I brought this up to my roommate again after the fact, and she just kind of shrugged it off (she's very unconfrontational, so I expected the reaction and he doesn't come around very often). A few years have gone by, and he's going to be coming over again. It never fails for him to make the rude comment about the blanket, and when he does I'm pretty sure I'm just going to explode and call him a piece of shit. He is not my friend at all, and he's the only friend of my roommate's that I really just can't stand and I've made this known to her whenever he comes up in conversation (rarely). Typically when I reach my tipping point I don't have much of a filter and that's what'll go down if anything is said.

Would I be wrong for blowing up on my roommate's friend?
```

**Verdict:** KEEP

---

## 204 · `637b7efaeca53a06` · 1000c · ask=body+title

> **Title** (not part of the prompt): WIBTA for decreasing a groups grade?

```text
For context, I'm currently in a statistics course in which, for the final project, we have to go and collect data and use it effectively. All data collection must be random and this is on the rubric.

A few days ago I was asked to fill out a survey about a fairly contreversial topic for a group who we'll call Z from here on.

So i fill out this form honestly but the third question (open ended mind you) asks how you would fix the topic of interest. I write out that I do not believe that this topic is an issue and list a few reasons why.

I give Z the form back and after a few minutes they come back and tell me to keep the survey because of what I wrote for the third question.

Now to the assholeish part. Since we all have to give a presentation on our results at the end and are graded by our peers, I plan to give them a shitty grade for data collection and ask questions to "expose" Z in front if the whole class.

Am I taking this too far?

Would I be wrong for decreasing a groups grade?
```

**Verdict:** KEEP

---

## 205 · `6414993345ab0c59` · 475c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked a girl out, that my best friend happens to be crushing on?

```text
For context, me and my friend, (we'll call him Bill) are both 23. We lead pretty standard lives. Every weekend Bill and I get Qdoba, it's our favorite restaurant. And theres a girl who works there who is genuinely good looking. And Bill is really attracted to her. Problem is, Bill isnt the type of guy to ask a random girl out. But I most definitely am. (On mobile sorry for formatting)

Would I be wrong if I asked a girl out, that my best friend happens to be crushing on?
```

**Verdict:** KEEP

---

## 206 · `641919185a4f01b2` · 1239c · ask=body+title

> **Title** (not part of the prompt): WIBTA for not helping my co-worker?

```text
It's a 2 man team. We used to have very defined roles. I am the leader of the team. My job is more general, strategic and creative, while hers is fairly mundane. 

Recently she made some improvements to her output, which increased the time needed to do it. And at one meeting where management commented on her output, resulting in her having to use EVEN more time to do it. So she asked if i could take over a process, which took about 1 hour per week. I was taken by surprise and pity came over me so i agreed. On hindsight i shouldn't have. 

Ever since then, she had been trying to push more and more of her jobscope to me. It would be fine if it was a once off favor but she is basically trying to offload a few regular processes to me. 

I am responsible for the results of the team and if she is tardy, we cannot hit our KPI, i'm the one who has to face the music. 

I just can't stand having to constantly reject her request to offload regular processes to me. When i reject her she tries to argue and finally end off with "k fine i'll do it" when it was her jobscope to begin with.

Should i suck it up and help her to achieve overall results or just her her be to be fair to myself?

Would I be wrong for not helping my co-worker?
```

**Verdict:** KEEP

---

## 207 · `6433979c0c2fcd9b` · 713c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I asked roommates to spilt the bill of household items

```text
To start off I live in a house with 4 roommates. We split the bills evenly with me managing all of them and sending requests as needed. I set up a chore rotation to keep the house clean that 2 roommates mostly ignore, 1 of them barely uses the common spaces so its not a big deal. Only 2 of us have regularly bought household goods like dish soap, toilet paper, cleaning supplies, etc. Neither if us has asked for compensation for these. But for larger purchases, like heavy curtains to block the coming winter chill and keep our heating bill down I would prefer to have everyone contribute since it will help keep everyone's bill down.

Would I be wrong if I asked roommates to spilt the bill of household items?
```

**Verdict:** KEEP

---

## 208 · `64fef5fc77881433` · 884c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I asked my very loud roommate to quiet down?

```text
Currently, it’s 10:30pm. She is cooking and she is very loud. She likes to play music or podcasts while she cooks, or talk on the phone. She is usually in the kitchen for several hours. I’d be lying if I said we got along. She’s repeatedly ignored my attempts at conversation and friendship. Anyway, we don’t like each other much, I don’t think. That being said, I’ve always done my best to be considerate in my noise levels. I am sometimes out in the living space later at night but I am always very quiet. I try to be on a pretty strict schedule for work reasons, and I need my body to be used to being asleep and awake at certain times. The noise (primarily the music/podcasts/phone calls, and the banging of pots and pans) prevent me from sleeping. Would it be terribly unreasonable of me to ask her to quiet down?

Would I be wrong if I asked my very loud roommate to quiet down?
```

**Verdict:** KEEP

---

## 209 · `66f7b2ea80e13836` · 1236c · ask=title

> **Title** (not part of the prompt): WIBTA for not helping this girl?

```text
I worked with a girl, L, on a uni project. Throughout the project we had a weird relationship where sometimes we'd be friendly, but at other times she would be rude to me. For example, she would tell me that I was working really slow, everyday she would hover over me when I was working and remind me to do everything I knew how to do. She would also send coursemates to me with reminders to do stuff, and prevent me from doing certain things because I'd do it too slowly apparently. I put together a tube of reagents we both needed for a future experiment and she watched me do it only to say she'd prefer if we'd used one she'd done. I have dyspraxia so I do things a little bit slower but it doesn't mean that I did anything poorly. 

Anyway, we're doing the write-up now and she keeps messaging me asking for help. She said she wants to share ideas with me. However, I know what I'm doing, I kinda feel like that's her problem and I don't want to risk being seen as colluding. I feel like I might be TA though because this is an important piece of work and we're getting close to a deadline, but she's stuck.

**tl;dr: Girl who I had a poor working relationship with now wants my help.**

Would I be wrong for not helping this girl?
```

**Verdict:** KEEP

---

## 210 · `67213cba439180c2` · 1309c · ask=title

> **Title** (not part of the prompt): WIBTAH if I complained to someone about a coworker who loudly listens to music?

```text
She wears headphones, but its so loud that other people say they can hear her music over their own. I'm not big on listening to anything at work, so it gets to me the most. She also taps her foot and sings sometimes. Now, if she was anyone else, I would ask her nicely to turn it down. This woman is not approachable at all. She is unfriendly, defensive, and rude. She had a fit when someone called her out for slamming her mouse in anger. She still does that... about 3 times per day. I have three options:

1. Deal with it. (Its very distracting. It would just continue to bother me until I eventually cracked during a rough day and said something I would regret.)

2. Go to her superior and calmly explain how distracting it is. (He would probably slip up and say it was me. And I dont want a rat reputation.)

3. Call the anonymous HR hotline. (She would get a slap on the wrist, hopefully stop due to the 'official' HR complaint, and no one would know it was me.)

I'm leaning towards options 1 or 3. But WIBTAH if I called the hotline? I know it bothers others in the vicinity, but not enough for them to call since they all use headphones. She wouldn't get written up for something like that, just a talking to.

Would I be wrong if I complained to someone about a coworker who loudly listens to music?
```

**Verdict:** KEEP

---

## 211 · `672c35eb54b8de2d` · 1742c · ask=title

> **Title** (not part of the prompt): WIBTA if I left my friends house early because of their dog?

```text
I'm supposed to be staying at a friends house for five days. I'm on day two and I'm really freaking out. To preface this story, I must say that I have some slight obsessive tendencies and I really struggle with a lack of cleanliness.

My friend and her mum have a tiny little terrier who's quite elderly and doesn't have the best control over its bodily functions. In the last 24hrs I've dodged multiple piles of poop and piddle and while they do their best to clean up, there's smelly stains and old streaks of dried poop everywhere. This dog will shit everywhere and then come and sit on me and lick me and lick everyone and everything and then we all sit down to share food and they haven't washed their hands and it's all a bit too overwhelming. I constantly feel sick and want to scrub my whole body down and change my clothing which of course is impractical as I have limited clothing on me.

I am well known to them as an avid animal lover so I can't just ask them to keep the dog away from me as it'd be considered out of character. And her mum's self concious enough about her house at it is without me being honest about why I'm leaving (if I do) so I was considering saying I'm feeling unwell and training back to mine (I live a 2hr train journey away). There's just no way I can be polite about it (or so I can think).

I fear I'm maybe the arsehole because I have exams coming up and I accepted the holiday invitation out of politeness. I do just want to be at home to cram, and this is a factor in my desire to leave, but this situation is really feeling like a dealbreaker for me. As much as I care about my friend, i want to make my excuses and leave...

Would I be wrong if I left my friends house early because of their dog?
```

**Verdict:** KEEP

---

## 212 · `68bb47e407ba458c` · 1685c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA For not letting my friend donate a kidney to me

```text
I have a rare autoimmune kidney disease that recently progressed to the point of end stage renal failure. I'm in the process of getting evaluated for a transplant, and my first appointment with the transplant centre is next week. After this, any potential donors can come forward and offer to be tested for a living donor transplant. The plan right now is for just my parents to be tested. One of my friends is adamant that she wants to be tested to see if she's a match, and if she is a match she wants to be the one to give me a kidney. She's an extremely stubborn and driven individual and has very firmly made her mind up on this. We also know already that we're a blood type match, so there is a solid chance that she would be a potential donor. The thing is, I don't like the idea of her donating for a couple of reasons. First off, the condition I have has a high rate of reoccurrance in transplants, so theres a strong chance (between 30 and 50%) that my body will wreck a new kidney and I'll be back in renal failure in a couple of years at best, so she could just be wasting her kidney on me. Secondly, it's a really invasive procedure with a long recovery time, and I know she wants to go travelling and explore the world, and I don't want to be the thing that holds her back from that. A couple of my friends and family don't see things from my side and think I'm either being too stubborn to accept help, or too pessimistic about my outlook if I do get a transplant, and think I should let my friend get tested. We've also argued a couple of times over the situation and she thinks I'm being unreasonable.

Would I be wrong for not letting my friend donate a kidney to me?
```

**Verdict:** KEEP

---

## 213 · `68e1022f5082759b` · 791c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I criticize an answer to an Amazon product question?

```text
Here’s a real petty one for y’all. I am so tired of people answering product questions on Amazon with a literal “I don’t know” instead of just ignoring the question. I was shopping for fabric cube-drawer-things for a modular, DIY cube shelving unit. The fabric cubes were by the same brand as my shelf, but pictured inside a different shelf, so I asked if they would fit in the DIY shelves (after double- and triple-checking what the shelves I bought were listed as). Someone replied with “Just measure your shelves! Also they’re nice and look great! Hope this helps!” (the company that produces the shelves later replied and said yes, they’ll fit). Or should I just roll my eyes really hard and ignore it like usual?

Would I be wrong if I criticize an answer to an Amazon product question?
```

**Verdict:** KEEP

---

## 214 · `69397dcd426c63e7` · 875c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told my friend I didn’t want to be friends anymore?

```text
TL;DR I play PS4 and met this guy who I was friends with for a long time, then he left for a couple months, now he wants to be friends again but I don’t want to because I realized I like being alone. So this starts about April of last year. I met this kid on Overwatch and we hung out for a long time. Fast forward to November and he says he doesn’t want to be friends. I understand why and we just both move on. From then to about a week ago I played alone and I actually liked it. Then last Monday he messaged me saying he wanted to be friends again and hat he felt bad for just leaving and I agree and we hang out. It was fine but it didn’t feel right for me. I want to tell him I like being alone but I don’t want to seem like an asshole. (Also sorry for any spelling mistakes I’m on mobile lmao)

Would I be wrong if I told my friend I didn’t want to be friends anymore?
```

**Verdict:** KEEP

---

## 215 · `69ba805977b95392` · 722c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked my flatmate to sweep their pubes in the bathroom?

```text
Now, I know in a shared bathroom I should expect some hair and some pubes, it's just how it is. But the thing is there are only three people using this bathroom, flatmates A and B, and me. I know the pubes are not mine, or flatmates A's. Regardless, it's like a pube-nado ran through the bathroom. My feet are often turned into impromptu hobbit feet after visiting. Well that may be a bit of an exaggeration, but my feet are the donut to the 80s bush sprinkles. Gross. What's even weirder is sometimes it's hair and pubes, but they have all been swept to the centre tile of the bathroom. Or am I over reacting and this is just how flatting is?

Would I be wrong if I asked my flatmate to sweep their pubes in the bathroom?
```

**Verdict:** KEEP

---

## 216 · `6a60f6b506626159` · 629c · ask=title

> **Title** (not part of the prompt): WIBTA if I went to a wedding instead of a funeral?

```text
Monday my grandmother's husband passed away and the funeral is planned on the same weekend that my friends are getting married on.

I've known my grandmother's husband for basically my whole life. I've never bonded with him but I also have never had any trouble with him. I only want to go to the funeral to be there for my grandmother.

On the the other had I already bought plane tickets and a hotel room a month ago for the wedding. The total price was about $350. I also agreed to help pay for the bachelor party. I can't  get refunded for the tickets and hotel.

Would I be wrong if I went to a wedding instead of a funeral?
```

**Verdict:** KEEP

---

## 217 · `6b02921d5ceea33e` · 743c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I posted an amazon wishlist on the event page for my daughter's first birthday?

```text
I don't want anyone to feel like they HAVE to get her anything, but if people do happen to want to bring her a gift, I really just don't want her to get a bunch of stuff she doesn't need. I don't mean for that to sound ungrateful or shitty or anything. But she has a lot of toys already and she really doesn't need 50 stuffed animals. I only have so much room and there are tons of things she actually needs with the warmer weather coming up. I have items ranging from 5-6 bucks, to my most expensive item being 30. But most things I would say are around 15 dollars. Would posting this list as a guide/gift idea type thing make me an asshole?

Would I be wrong if I posted an amazon wishlist on the event page for my daughter's first birthday?
```

**Verdict:** KEEP

---

## 218 · `6b1d7f078d8867e5` · 1593c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I ask my personal trainer to stop working out during our sessions?

```text
Non-native English speaker and first-time poster, so bear with me please. I've started training 3x/week with a personal trainer, as I desperately want to lose weight. This has been going extremely well, even with my nervousness beforehand. We get along well and I really felt like I was improving. We even became friends in a way, which may have been a mistake. He's a lovely guy and obviously knows what he's doing. However, a few weeks ago he asked me if it'd be alright if we trained together, assuring me I would get the same level of workout I would normally get. In the beginning I also really enjoyed the working out together, but now it's gotten to this point that he trains more during this hour than I do. (For example, we alternate on the same exercise with one person doing it and the other person waiting. Now he always does 4 sets and I do 3 or even 2 with me waiting during his 4.) As I'm paying a lot of money for this - and as nurse it's a real big part of my paycheck - I want to ask him if we can go to our usual workout where he just supports me while I'm doing the exercises. However, I fear it might be selfish of me. And he's really enthusiastic about it, defintely not rude. As a non-assertive person, I'm wondering if it's worth bringing it up, or if I should just keep going like this and be grateful for the workout I get. TL:DR; My personal trainer starting working out with me during my sessions, giving me the feeling that I don't get the same level of training as I used to.

Would I be wrong if I ask my personal trainer to stop working out during our sessions?
```

**Verdict:** KEEP

---

## 219 · `6bbf4d98f23910db` · 1868c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I told my best friend to leave her husband and move across the country?

```text
My friend is currently in the transition between jobs. She just quit working her job on the east coast and has one lined up  on the West coast next month.

She also got married recently, with in the last 2 months, I can't totally remember. Her relationship has been a bit bumpy from what she has told me. Nothing too crazy, but she says she isn't happy where she is in life and this job is really going to help her since she will be living in the city of her dreams doing a job she loves. 
This means that her husband also needs to move across the country too. It appears that he is able to and still keep his job working remotely. 

Recently she has made it clear that she and her husband were having problems and that she doesn't feel like he values her and she is treated like anyone else. Ling story short, she talked to him about it and she called me crying and saying that he was leaving her. 

The options came down to her staying in a city she hated, jobless, and her possibly working through her relationship to fix their problems, but still no promises, so let's add alone to the list. Or she could move across the country to a city where she has some family (although her relationship with them is a bit rough), a stable job.

I'm her best friend, she doesn't really have a lot of people beside her husband, me, and a couple others. I'm feeling useless and I dont know what to tell her. I want her to be happy, but I don't want her to be alone across the country. She is only 20 and has her whole life ahead of her. I told her to take a couple days and see how she felt then about everything as her judgment could be clouded right now. I personally want her to be happy and am learning towards her going to her new job, but I still would like some others opinions.

Would I be wrong if I told my best friend to leave her husband and move across the country?
```

**Verdict:** KEEP

---

## 220 · `6bf9cf9705b762c9` · 1730c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I kicked out my brand-new housemate because he's not clean enough?

```text
We (2 F) just got a new housemate a few days ago and I'm already wishing we hadn't chosen him. So, we have a 10 month old baby who crawls everywhere and has her fingers in everything. Our new housemate, G, is a super lovely guy but A. He doesn't wash his hands after going to the toilet (something I've tried hinting at to avoid addressing it directly) and B. He spent 45 minutes in the bathroom trimming goodness knows what hair on his body (I saw him doing his belly & later he closed the door which can only mean pubes, right..?). I left before he finished, and when I got back (He wasn't in) I wanted to do the baby night routine when I found his hair everywhere!! On toothbrushes, contact lens cases, mouth guard etc. When I saw loads floating in baby's bath, I lost it and texted him. I was frank but nice (I asked a friend). While I was at it I mentioned the hand washing. He is also a smoker despite us clearly stating non-smokers only. (He told me he assumed outside is ok. No, it's not, man!) I THINK he now vapes in his room (which I'm not happy about) but he says it's without the nicotine. Either way, his room smells stuffy already. (It drafts into the hall.) Oh, and I also went on a cleaning spree after this and I found parts of the toilet full of pee spray. I'm seriously grossed out. But he is a nice lad! (I haven't talked to him yet because he's still not home but I intend to in the morning. I talked to him about the smoking and he said "I'll stop. Dont worry." But I know from someone who saw him the next day that he had a cigarette.) Would that be unfair? Or should I give him a chance and see if he betters himself?

Would I be wrong if I kicked out my brand-new housemate because he's not clean enough?
```

**Verdict:** KEEP

---

## 221 · `6c4657eb80f74e97` · 1759c · ask=title

> **Title** (not part of the prompt): WIBTA if I told someone to stop sitting with me because they are annoying.

```text
So I am normally not annoyed by anyone and am typically very accepting of new friends and new people. But at the beginning of this school year this girl just showed up and asked to sit at the table am sitting at. I said sure, as I really didn’t mind at the time. Now I sincerely regret it.  She immediately started treating me as if we were best friends and she was over sharing her entire life story with me. It made me uncomfortable. She touches me a lot when I’m obviously uncomfortable, and I’ve even directly told her to stop touching me but she just won’t. I’ve tried to just tune her out and tolerate her- but if she thinks I’m paying to much attention to my phone she literally steals it from me and won’t give it back till I pay sufficient attention to her. She will also shove me randomly as a “joke”. I’ve told her so many times to stop stealing my stuff and shoving me. I’ve even gotten so mad I’ve yelled at her but nothing gets through her head. Anytime I tell her to stop she just goes “okay, you don’t have to act like my mom you idiot” and laughs. It’s not funny. Everyone in my friend group does not like her and we are all uncomfortable. 

So I really just want to be direct and tell her to honestly just fuck off. Like I want to explain that we are not friends, and she needs to stop talking and sitting with me. Would that be to harsh? I don’t want to hurt her feelings but I’m so uncomfortable and irritated around her. 

TL:DR: I let a girl sit with me at lunch, now she thinks we are best friends and she oversteps her boundaries and steals my phone. I want to tell her to leave me alone and stop sitting with me but I don’t want to be an asshole.

Would I be wrong if I told someone to stop sitting with me because they are annoying.?
```

**Verdict:** KEEP

---

## 222 · `6cb7ec6796054749` · 919c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I were to ‘lecture’ people every time I see someone being an asshole?

```text
This is gonna be kinda rhetorical. So I live in a kinda place where people really don’t follow social rules all that much. Everyday that I have to communicate with the general public is a day I get upset. For instance, people cutting in line, people playing stuff without headphones, people just being straight up rude, yelling, arguing in public, (sales) people discriminating based on weight, race or gender and so on and so on. Sometimes I say something if I have an energy for an unpleasant situation and sometimes I just try to ignore these things. I have this fantasy where every time I see something like that I would tell the person that what they are doing is wrong and then proceed to explain exactly why it’s wrong and what they can do to improve themselves (and thus the society, making things better for all of us).

Would I be wrong if I were to ‘lecture’ people every time I see someone being an asshole?
```

**Verdict:** KEEP

---

## 223 · `6d62784a8bc1f71a` · 1150c · ask=title

> **Title** (not part of the prompt): WIBTA if I took my cat from my dad and his cat who she's bonded with?

```text
TL;DR My dad has been watching my cat for the past year and it's time that I can take her back, but shes became good friends with my dad's cat. He said I cant take them both if I want, but he lives alone and I know that he doesnt actually want me to take his cat. Would it be a dick move to separate them?

About a year and a half ago, I got a cat(Luna). I love her more than anything and I'd do anything to make sure shes happy. I didnt find out until after moving to my current place that my roomate was allergic, and I had to send her to live with my dad for the year. My dad ended up getting his own kitten(Paris), who has grown up with Luna and now they're really good friends. My dad says I can take them both if I want.l, but knowing him, I know he would be really sad if I took them both and hes just trying to spare my feelings. At this point Luna has lived with them more than me and I'm not sure if itll just be better for me to leave her there. She seems really happy there right now and I'm not sure if itd be selfish of me to take her back at this point.

Would I be wrong if I took my cat from my dad and his cat who she's bonded with?
```

**Verdict:** KEEP

---

## 224 · `6e179187bb473d78` · 614c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I bail on someone who asked me to prom?

```text
So a couple of days ago a girl asked me to prom. We’re both juniors and she seemed pretty excited, but hasn’t shown any interest in me recently and we haven’t talked much, so I believe we’re just going as friends. There’s a senior girl who was going to ask me to prom, but the junior asked me first and it got kinda complicated. I’d much rather go with the senior, as I think we’re both interested in each other and it’s her last year, but I don’t want to make the junior feel bad by bailing. The prom is in two months and she asked me about a week ago.

Would I be wrong if I bail on someone who asked me to prom?
```

**Verdict:** KEEP

---

## 225 · `6f83f6bfef07813c` · 1115c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I broke a diet for myself but not for a friend?

```text
Basically I am intolerant to gluten (whether intolerance or celiac is unknown) and when I eat gluten I usually (but not always) have a very upset stomach and diarrhea for the next 4-6 hours. So because of that I eat very strictly gluten free.

Whenever I eat out with my friend we have a limited selection of where we can go because of my diet. She doesn’t eat gluten free at the places we go, but we try to go to places that serve gluten free.

However I recently found out that my favorite gluten pizza place is only two blocks away from my school, and even though my stomach would feel horrible afterward I still want to eat there just once as a treat.

I’m conflicted because on the one hand I’d really like an actual slice of pizza just once, but in the other hand I feel like I’d be a major asshole if I break my diet for myself but not for my friend (by that I mean go to a gluten restaurant with them). Another thing is I feel like if I break my diet it makes it seem like it’s not so bad so I may be asked to do it again.

Your thoughts?

Would I be wrong if I broke a diet for myself but not for a friend?
```

**Verdict:** KEEP

---

## 226 · `700f3e81e38fdda4` · 605c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for not wanting my friend to bring another friend Comic-Con

```text
So to start off I'm bringing my friend to a very small Comic-Con for her birthday. We had planned to go together as a pair, but she wants to bring another friend. I don't know this person at all. I don't know what they look like or anything. My friend has a tendency to stop talking to me in favor of talking to other friends, when it's just us hanging out. She took a call with internet friends when we were at the pool. Anyway I feel like I should let her take them, but at the same time I don't want to be left in the dust.

Would I be wrong for not wanting my friend to bring another friend Comic-Con?
```

**Verdict:** KEEP

---

## 227 · `710b620dcc773600` · 802c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I finessed the best dorm room for next year?

```text
Some background: At my college, students must stay on campus for their first 3 years and have the option to move off for senior year. Also, almost everyone stays in the dorm for all years they live on campus. I am a current freshman applying for housing next year. So here's my dilemma: I have ulcerative colitis, an auto-immune disease that affects my colon and other pooping aspects of my life. There is exactly one room with a bathroom in it in my dorm. The seniors who stay on campus have priority in picking this room. Would I be an asshole if I used my condition to get picking priority and essentially finesse the best room? TL;DR: I have a condition that could allow me to take the best room in my dorm that seniors usually get.

Would I be wrong if I finessed the best dorm room for next year?
```

**Verdict:** KEEP

---

## 228 · `710c35dbc8866ba5` · 426c · ask=title

> **Title** (not part of the prompt): WIBTA If I purposely advocated for the least popular consensus in these posts?

```text
I mean, it's great to get unanimous NTAs or YTAs, but do you think it would be productive to argue from the other perspective? OR would this just be considered trolling? Because in every situation, there's never a clear asshole. We only ever hear the perspective of the poster, which more often than not, is made to flatter themselves.

Would I be wrong if I purposely advocated for the least popular consensus in these posts?
```

**Verdict:** KEEP

---

## 229 · `7156db4890d34bdb` · 585c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for confronting a coworker about my name?

```text
My legal name is different than what I go by. I really don't care which one people use and I let people know that. Now I have this co-worker who constantly will call me my legal name, then "correct" herself to my nickname. I have told her probably 4 times now that I do not care, at all, which name she uses. But it's starting to get weird that she keeps correcting herself. I don't really know why I feel that this is weird, but it makes me uncomfortable when she corrects herself like my legal name is "wrong" or something.

Would I be wrong for confronting a coworker about my name?
```

**Verdict:** KEEP

---

## 230 · `71d50f2f50d556be` · 1237c · ask=body+title

> **Title** (not part of the prompt): WIBTA for asking someone for half of the bill for vehicle repairs after an accident.

```text
Earlier this week I was leaving work and another person at work who I'm very good friends with backed into me and caused enough damage to my car to rip off my bumber and destroy most of my back drivers door ( still opens and closes fine just broke the panel and caused a huge dent). We both drive late 90s vehicles and only have liability. We decided not to file a report because then work would get involved and it since my car would be deemed totalled insurance would most likely want it in exchange for a few hundred bucks. I'm only worried about replacing the bumper and lights because I have to since it holds my license plate in place, the back door I dont care about. Right now its looking like I'm going to be paying 500-700 hundred between replacement and someone to put it on correctly. His truck has absolutely no damage just paint from my car. Hes saying I ran into him when its clear from everyone who witnessed it that he was backing out and not paying attention. Should I just eat the cost and replace it myself or ask politely for half the cost of repairs ( I would gladly pay half of his bills if the roles were reversed).

Would I be wrong for asking someone for half of the bill for vehicle repairs after an accident.?
```

**Verdict:** KEEP

---

## 231 · `7311467fa463cce5` · 646c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I got mad at my sister for breaking a gift she got me?

```text
My sister got me a nice contour make up palette for my birthday. The next day she asked if she could use it and I said yes since I didn’t mind. She dropped it after using it, making the make up inside crack. She didn’t even tell me she broke it, she left it until I opened it later and then asked her about it. 

 I’ve been using it since bc I don’t have the money to buy it myself and it’s a really nice one, but it’s really hard to use and I usually end up making a mess. 

Would getting angry about it make me ungrateful since she bought it for me in the first place?

Would I be wrong if I got mad at my sister for breaking a gift she got me?
```

**Verdict:** KEEP

---

## 232 · `73457fd37661098c` · 522c · ask=title

> **Title** (not part of the prompt): WIBTA for asking my friend's widow for his writing materials.

```text
So my friend and I were both writers (nothing professional or anything). And he very suddenly passed away in his sleep. After processing what happened, I got this itch to get what material he had written for this fantasy novel he was working on and finish it for him in his memory. However, I want to give his wife (who I don't know well at all) some space to grieve and I don't want her to feel like I'm taking advantage of her husband's passing.

Would I be wrong for asking my friend's widow for his writing materials.?
```

**Verdict:** KEEP

---

## 233 · `735b6dc03969c209` · 735c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I don't invite my lonely friends over for the holidays?

```text
I'm really torn about this one. One of my friends is this guy who lives somewhat by himself, and the other is  19 who JUST moved to my city. I've introduced them to each other and they seem to have their own budding friendship. I live with my parents, and they are very big on NOT being hosts for the holidays as they like to relax and kick back without having anyone over. It just occurred to me that they may be lonely tomorrow and I feel responsible for them both because I usually check in with them and hang out and whatnot. I don't want them to be lonely on Thanksgiving but I also don't want to put my parents under unnecessary stress tomorrow either.

Would I be wrong if I don't invite my lonely friends over for the holidays?
```

**Verdict:** KEEP

---

## 234 · `73c2fae961f04c3d` · 556c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I went into a stall if other people are waiting, when those other people didn't check to see if the door was unlocked?

```text
Long title, but that's basically the entirety of it. Our work bathroom has 4 stalls (3 reg, one handicapped). They actually have doors and no gaps, so if the door is closed, there's no way to visually determine if someone is in it. People will frequently leave the stall after using it and close the door behind it, which makes incoming users think it's occupied. Even if multiple people have been waiting before me?

Would I be wrong if I went into a stall if other people are waiting, when those other people didn't check to see if the door was unlocked?
```

**Verdict:** KEEP

---

## 235 · `74af08a0b9bfc934` · 1164c · ask=body+title

> **Title** (not part of the prompt): WIBTA for taking the easy way out of a possibly sinking ship?

```text
I'm currently working at a small company, not for long, somewhere around a year now. The work environment is really nice, I'm quite good friends with almost everyone. Problem is it doesn't pay that good (not that bad either, a little below average) and it's unlikely to change in predictable future. That's not the biggest deal-breaker for me tho. The problem is that company is struggling right now, there are new possible projects, but if they fail (there is a good chance for that) we are screwed. I took that gamble before with another company and it backfired horribly, so I'm not eager to take the risk again.

Recently I got a possible opportunity from a bigger, more stable company with high-ish chance for bigger paycheck.

My dilemma is that if I take it, it will probably ruin some future projects. I'm not THAT important that my leave will instantly collapse the company, but losing 25% of your workforce is going to hurt no matter what. It will inevitably lower the chances of company surviving.

What do you think guys, would I be an asshole to my coworkers if I took the job?

Would I be wrong for taking the easy way out of a possibly sinking ship?
```

**Verdict:** KEEP

---

## 236 · `74b297d6c8b4b798` · 1494c · ask=title

> **Title** (not part of the prompt): WIBTA for sending out a group message to all my coworkers telling them to make sure movies are scanned in as to avoid 'Not scanned in's'

```text
Basically since these three new hires were put on the team last month we've been having a HUGE problem with Not scanned in's or NSI's so far. I've pulled at least 15-20 movies off the shelves that weren't scanned in and some so late they were about to become purged (basically when a movie is kept out for more than 25 days we put a fee on the customers account on top of accumulated late fees that they have to pay back before they can rent again.) So now people are getting late fees and possible purge payments that they shouldn't have got. I wouldn't assume that it wasn't the new employees who were doing this if 

1. A movie that my dad had rented out under my account and returned yesterday was sitting out on the counter when I opened this morning, not scanned in. The two people who were working yesterday was 2 of the new employees

2. This problem didn't start happening until they got hired. We had maybe 1-2 NSI's every couple of weeks before the newbies were hired. 

I want to do it as a group message to all my coworkers so none of them feel targeted or feel like I'm singling them out as I don't want to cause drama. But the NSI's are getting quite bothersome because it really pisses people off and makes them lose trust in our business. Which in turn can cause us to lose business! 

*tl;dr** newbies are harming the business

Would I be wrong for sending out a group message to all my coworkers telling them to make sure movies are scanned in as to avoid 'Not scanned in's'?
```

**Verdict:** KEEP

---

## 237 · `74cbc6387e8dc7ff` · 1240c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for asking my grieving friend for help?

```text
A little backstory, my friends cat died on Monday and she hasn't taken it well, that cat was her best friend for a long time. He (the cat) was only about five and very Healthy and street smart, (she let her cat's outside) she was not ready at all, her nextdoor neighbor is an complete asshole, he let his dogs run around without a leashes in an unsourced backyard, this neighborhood has a lot of cats he knows this, so Monday his dog got out and chased my friends cat onto the road it was hit by a car, and sadly passed and all the guy did was half-heartedly say "oh sorry my dog got out" my friend texted me telling me what happened, at first she seemed deactivated, but today the day after, she's acting like nothing happened like everythings normal again, and that hurts that cat can't just be forgotten, well today it all caught up with me you see I spend alot of time at her house, and I loved that little guy as if he was my own I'm not taking it well at all, I was there right before he passed and just can't accept it, I would go to other people to talk it out, but she's the only one of have in my life right now, and she seems to want to just forget him and what happened,

Would I be wrong for asking my grieving friend for help?
```

**Verdict:** KEEP

---

## 238 · `751a330028141324` · 1376c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA If I reported my downstairs neighbors for smoking hella weed?

```text
So yeah, sorry this won't be super juicy or interesting. The situation is pretty simple and straightforward. My downstairs neighbors like to turn up. Now, I dont care about weed. I believe it should be legal, medically and recreationally. I've had many friends that partake in the devil's lettuce. However, I live in a state where it is very much not legal. It's not even decriminalized. You can get in a lot of trouble out here for it. Also, I live in an apartment. My apartment has checkups every so often by maintinance to check smoke detectors and air filters. I don't want to end up getting raided or something because a maintinance guy catches a whiff of the green stank that's finding it's way into our apartment. Maybe I'm worrying too much. Maybe it doesnt matter. But just the other day, coming home from the store, I swear I could faintly smell weed in OUR ENTIRE APARTMENT. Maybe we have a connected HVAC or something, who knows. I know they do it on the balcony, and it woffs up to our apartment through our door. Should I tell maintinance/landlord? Should I tell the authorities? Would I be putting someone in a lot of hot water over something trivial? Tldr; The people living under us smoke a lot of weed, and it is making our apartment spell of said weed. Should I just let it go?

Would I be wrong if I reported my downstairs neighbors for smoking hella weed?
```

**Verdict:** KEEP

---

## 239 · `7554db157b502444` · 1744c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I quit my group?

```text
So I’m in a quartet of high schoolers that formed spring of last year. I was actually invited to join them, as a former member had just quit and they needed another for some competitions they were planning. I was never supposed to be a permanent member, but because we won all the competitions,  parents decided to keep it going into the following fall semester. The four of us were all going to be seniors, so I was a little hesitant to continue, knowing that college apps, and classes were going to take up a lot of time. Fast forward to now, I’m very heavily considering just quitting the group altogether because of the flakiness in some members. For example: The entire quartet would agree on coming in an hour early to work on our pieces. However, half of the quartet didn’t show up, and I was left wasting an hour of my time. The same half would also often come to rehearsal late (~15 mins) The same half would come to a performance with only ten minutes to warm up, when we all agreed to have an hour. Rehearsals not led by a coach would be unproductive. I have tried talking not only with the members, but to the parents themselves, but have seen no improvement of being on time. I can’t be mad at any of the members, because it would ruin the entire rehearsal (quartets are all about communication) Ive talked about quitting numerous times with my mom, citing the blatant disrespect, and the amount of time wasted on my weekends (~3 hours), but she thinks it would be an asshole move, considering I already agreed to commit last fall, and we have performances planned. As much as I understand my mom, I can’t help but still feel justified, considering I didn’t know about the flakiness last fall.

Would I be wrong if I quit my group?
```

**Verdict:** KEEP

---

## 240 · `755b16ff80026474` · 352c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for not talking with my friends?

```text
So, a few months ago, I started talking to some old friends I haven’t talked to in a pretty long time. I noticed that I always contact with them first, unless they’re asking for favors. I don’t really like this, to be honest. I don’t want to be the only person trying to maintain the relationship. So.

Would I be wrong for not talking with my friends?
```

**Verdict:** KEEP

---

## 241 · `7580b1d21c455164` · 791c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I didn’t switch parts with my understudy on his birthday?

```text
Not much else to add. So, on our opening weekend for my school’s theater musical the Friday we open is my understudy’s birthday. He’s turning 18, and asked me about switching our performances so I would perform on our opening Saturday and he on our opening Friday. Now, the understudy for my role’s best friend isn’t a great actor, and also hates my guts. Our faculty advisor for the show said it would be ok if we switched shows, but won’t let my role’s best friend’s actor switch performances with his understudy. I told my understudy I wasn’t sure if I’d be okay switching performances, and he said that’d be a pretty shitty thing to do, especially since I’d basically promised I would switch, which I hadn’t.

Would I be wrong if I didn’t switch parts with my understudy on his birthday?
```

**Verdict:** KEEP

---

## 242 · `75d51e98fee0fd2d` · 581c · ask=title

> **Title** (not part of the prompt): WIBTA if I went out of my way to tell this girl that shes being cheated on.

```text
Ill spare the details. I do not know this couple, they are a friend of a friends. All I know is that they are engaged and the guy has been cheating on his girl with one of my close friends and allegedly multiple other women. On one hand I feel like since I know nothing about these people I should stay out of their business, but on the other hand I have been cheated on and if I were in her situation I would be grateful if a stranger provided me with this information for me to investigate.

Would I be wrong if I went out of my way to tell this girl that shes being cheated on.?
```

**Verdict:** KEEP

---

## 243 · `767f44174774e126` · 654c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I had taken a slot for a sports competition instead of someone else?

```text
At my school we have an inter-house sports competition, in which I wanted to fence, as it's the only sport I'm any good at. However, there were only 2 slots, and after I signed up, I found out that the other two serious fencers in my house wanted to go. In the end, I let them replace me in the slot.

However, it's been bugging me as to whether I'd have been the asshole if I kept the slot or not, bearing in mind that the other two are in the year up and both better than me by a considerable margin. (In fact, one's a GB fencer.)

Would I have been the Asshole?

Would I be wrong if I had taken a slot for a sports competition instead of someone else?
```

**Verdict:** KEEP

---

## 244 · `7717a6ad5dd204ee` · 1615c · ask=title

> **Title** (not part of the prompt): WIBTA if I commissioned someone else instead of my friend?

```text
I have a friend who we'll call Craig. We're relatively good friends (at least I think so), and they often help me with my problems. Craig has an SO who we'll call Kenny; Kenny and I had a falling out in the past and haven't interacted much since then, but what few interactions we have had seem a bit more friendly now. 

Craig and Kenny are both (very skilled) artists and writers, and Craig has given me advice on my own writing in the past. Another friend of mine, Brian, recently gave me some money as a gift. I decided I wanted to use it to commission some art from Craig and Kenny, and contacted Craig asking about it. They seemed happy and grateful, and said they'd talk to Kenny about it, but Kenny was sick at the time and so it took a while to get a response. A few days ago, I told Craig what I'd like the art to be of, and they responded, again, they'd check with Kenny to see if that art fit the money I could pay, though it would only be a sketch. I haven't heard from Craig about it again, though that is partly my fault because I didn't bring it up in conversation after that. 

Just today, I found another artist on Tumblr. They take commissions as well, and will do more for the same price. I'm considering commissioning them instead, but I would feel bad for canceling on Craig and Kenny when I already told them I'd be buying from them. 

**TL;DR Have artist friend, get money, want to commission him and his boyfriend, tell them that, don't hear back about it for a while, find another artist and want to commission them.**

Would I be wrong if I commissioned someone else instead of my friend?
```

**Verdict:** KEEP

---

## 245 · `77a1036432f67d9d` · 1427c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I (and my friends) only went to half of my friends birthday party?

```text
My group of friends have been close for about 5-6 years now. We all play video games, D&D, hang out and occasionally party together. A member of our group has his birthday coming up soon and will be turning 21. Now we're in a province where the drinking age is 18 and from 18 up through 19 we partied and went to the bar all the time. Now that we're all turning 21 this year we are pretty much done with clubs. It's loud, boring and way too expensive for the majority of the group. Except the birthday boy. He still goes out clubbing almost every weekend, and for his birthday he wants to do have a small get together and then go get wasted at the bar as per usual. He's a good friend, we want him to have fun, but the vast majority of us don't like to, or don't have the money to go out to the bar (some are both). We have a friend in from another country too, and he doesn't wanna go out. The birthday boy wouldn't be alone, he's inviting some other people and his girlfriend who loves going out with him. It's his birthday and we rarely get to see each other with school and work schedules. I feel like it may be rude to him if we all stayed at the house (it's not his the first half is at, it's one of our friends) while he went out on his birthday. But he knows we don't like to go out clubbing, and we really, really don't want want.

Would I be wrong if I (and my friends) only went to half of my friends birthday party?
```

**Verdict:** KEEP

---

## 246 · `78337c7624832554` · 1580c · ask=title

> **Title** (not part of the prompt): WIBTA for not wanting my my aunt and her husband at my graduation?

```text
I’m on mobile so, blah, blah... 

A bit of background. My aunt, who is like 70ish maybe lives in a different country than all of my family, so because of that, we’re not exactly close. Her husband, who’ve they’ve been married about ~35 years is an asshole and no one in my extended family likes him and the chooses to ignore this fact. She’s not a bad person, but she’s really annoying. Her family is super religious, really judgy and can’t stand when people (my whole family for ex) don’t go to church or have different beliefs than her. She and her husband are the literal definition of a conservative family... at my graduation party, my close friends and family are gonna be drinking and having an overall good time and they’re probably gonna be judging everyone for this... She has been asking me for months the date of my graduation, but i don’t answer her back. Just recently, my mom told me that she started asking her this also, but my mom knows that I don’t want her here, so she just tells her she doesn’t know. My aunt has a habit of unexpectedly coming over so I can’t tell her a date because she might come... 
Also, she always lectures me about god and how he’s the one that saves lives, not me or my colleagues (medical school graduation), so we would i want someone like that on such a special day? 

I’m sorry for grammar mistakes, english is not my native language. 

Tl/dr: annoying aunt and husband want to invite themselves to my graduation and I don’t know how to tell them off.

Would I be wrong for not wanting my my aunt and her husband at my graduation?
```

**Verdict:** KEEP

---

## 247 · `7a837467c06838b8` · 737c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for asking somebody to stop saying stuff like "God bless you" to me?

```text
There's this girl I chat with online sometimes. We're not close or anything, we just talk occasionally, and overall, she's really nice. She knows I'm an atheist, but almost every time we talk, she says stuff like "God bless you" or "I'll pray for you" or even something like "Hopefully you'll find Jesus soon!" She has good intentions, I guess, but it just bugs me that she keeps saying this stuff when she is fully aware that I'm not religious. The way I see it, if she wants to pray for me, she can do that without having to let me know every time we speak to each other. It feels like she's just disrespecting my beliefs (or rather, lack thereof.)

Would I be wrong for asking somebody to stop saying stuff like "God bless you" to me?
```

**Verdict:** KEEP

---

## 248 · `7aa6e035c3aa768b` · 770c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I told my sister she isn't allowed to use my mugs anymore?

```text
Some backstory, my sister (25) lives in a different state, and when she comes to visit, her and her husband drive several hot beverages, requiring mugs (I swear this isn't a shitpost). I have about 4 mugs that I cherish from a family vacation a few years back, and they are no longer made (aka not replaceable) and she uses them when she's here, and doesn't rinse them, and leaves horrid rings in them. So far I have lost one mug to nasty rings that ruined to glaze, so it's unusable. I don't want her to use them anymore. My mother doesn't want me to bring it up, because she thinks I'll disrupt the family peace. Also, I have told her numerous times to please rinse them, and she doesn't.

Would I be wrong if I told my sister she isn't allowed to use my mugs anymore?
```

**Verdict:** KEEP

---

## 249 · `7ba4e4c5ad5654b9` · 1441c · ask=title

> **Title** (not part of the prompt): WIBTA if I exclude one sister and take the other on our yearly vacation?

```text
I live pretty far from the rest of my family.  For the last three years I save up and take my sisters on a vacation.  My youngest sis and I are very close.  My middle sister not so much.  The relationship between the two of them is sometimes outright hostile.  

I’ve included her the last three times more out of obligation than anything else.  The reason I want to leave her out of this one is because she outright ruined the last.  She insisted on bringing her young daughter.  If me and my other sister did anything that a young child couldn’t be included in she would throw a hissy fit.  She kept starting fights with my other sister.  One of the nights she was letting her daughter be so freaking loud that the people next to us complained and hotel staff came to our room and asked us to quiet down.  Then she loudly said, “jokes on them, he didn’t even care”.  I’m pretty sure he wouldn’t have come to our room if he didn’t care.     Whenever her and my other sister would fight she kept threatening to leave early.  I just kept my mouth shut internally hoping she really would.  

These vacations aren’t cheap.  This is the one big fun thing I get to do all year.  There’s just always some drama or hassle with her.  I know it would probably hurt her feeling or piss her off or both, but I want to be able to actually enjoy the vacation this year.

Would I be wrong if I exclude one sister and take the other on our yearly vacation?
```

**Verdict:** KEEP

---

## 250 · `7ca7d363f90c67a3` · 1106c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for confronting my roommate about who does the dishes?

```text
Backstory: My lifelong friend and I (both 24M) recently moved in together after some life changes on my end. I (verbally) agreed to do the dishes as he hates it and I have (or had) no issue with them. This was fine for a few months, no problems. Recently I've been dining out a lot or just not been at home often and therefore have not been dirtying up many dishes. Those that I do use (and they are few and far between) end up washed that day/night or the following night (we do not make use of our automated dishwasher. His decision). My roommate uses a lot of the dishes, probably 1-2 plates a day and 1-2 drinking glasses every 2 days. The complaint: I do 100% of the dishes regardless of how often I am home or how many dishes I use. If I leave to visit family for the weekend, there will be a pile of dirty dishes waiting for me when I get back. This upsets me as I clean all that I use and he cleans nothing that he uses whilst I am absent. Am I just being childish and should I suck it up and do all the dishes as was agreed?

Would I be wrong for confronting my roommate about who does the dishes?
```

**Verdict:** KEEP

---

## 251 · `7cec97b5687af51b` · 725c · ask=title

> **Title** (not part of the prompt): WIBTA if I stop taking care of my Brothers Dog

```text
My bother randomly comes home with a puppy (pit bull) and I automatically I know he is going to dump the responsibility on me, my sister and my mother 

He almost never comes home only to sleep then leaves 

So he leaves the puppy alone for most of the day. 
And if I tell him that “I am not going to take care of it” he will say “then don’t”  knowing full well that I will give up and take care of it (It’s so painful hearing it cry) like Emotional Blackmail 

So will be an AH if I actually stop taking care of it completely?

In total I take care of it 95% and he just comes home and dose the 5%  which is basically nothing just bed time 

I am so stressed out 😤

Would I be wrong if I stop taking care of my Brothers Dog?
```

**Verdict:** KEEP

---

## 252 · `7df9ff53bba0a07b` · 1320c · ask=body+title

> **Title** (not part of the prompt): WIBTA for not going on our family holiday?

```text
So I (32f) am part of a very close family, we all love each other etc. Fam wants to go away for Christmas, to Fiji. I said no, I’d like to go somewhere new. One of my sisters has been guilting me for deciding to miss an opportunity to make memories with the whole family.

Backstory: I’ve been to Fiji three times with one sister, her family and my mum at least three times. I told them I wouldn’t go to Fiji during the planning stages, and left it at that, but my sisters kept planning and ended up deciding on Fiji despite my wishes, because they have kids and Fiji is easy with kids. I also like staying out, drinking and generally running amuck on holiday. Whereas my family is in bed by 9.30.

I feel like I might bet he asshole, because I have it easierand should be more flexible as a single female. Also because I’m willing to miss family time to explore a new place. But I also feel like I’m not the asshole because if it’s a family holiday, it should be a choice of all family members. 

My sisters are older than me and got to travel through their 20s, but because of the age difference during my 20s, my dad was sick so I didn’t travel at all. I’ve got the travel bug late, after dad passed away, while they’re busy having babies. 

So, judgment please?

Would I be wrong for not going on our family holiday?
```

**Verdict:** KEEP

---

## 253 · `7e299421b4d42b14` · 1033c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I sold hand me down clothes

```text
My cousin (whom is extremely well off) passed on children’s clothing to another cousin about 4-5 years ago. The second cousin then passed on clothing to myself and continues to do so about every 6 months. The original cousin has very nice but very expensive taste so all of the clothing we received are higher end name brand clothing and shoes. A lot of which still actually have tags on them even from being passed through 2 different kids before getting to me. Now I want to say, I am forever grateful that I literally never have to buy a single piece of clothing or shoes. But we have so much. I donate the pieces I know aren’t really for us and I’ve given some to a family in need but we’re still over loaded. I want to get rid of some of the clothing but I don’t want to donate it because I know someone will just turn around and sell it anyways. I feel really uncomfortable asking my cousin what I should do with the unwanted items because we really only talk like once a year.

Would I be wrong if I sold hand me down clothes?
```

**Verdict:** KEEP

---

## 254 · `7e9683070331de93` · 535c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I go to the movies while sick?

```text
A movie I bought tickets for months ago is playing this Thursday. I really want to see it, and this will be the only chance I get to see it, but I came down with a cold and I know it wont be gone by Thursday. If I go I will most likely be coughing a little bit as well as a sneeze or two. I'd feel bad for the people on both my sides as I already know the place will be full. I'd hate to go and feel like the people right next to me would be annoyed of possibly getting sick as well.

Would I be wrong if I go to the movies while sick?
```

**Verdict:** KEEP

---

## 255 · `7e9f15291bf6a3dc` · 644c · ask=title

> **Title** (not part of the prompt): WIBTA if I tell my mom to bring me to the doctor?

```text
Background: For the past 3 weeks whenever I eat large amounts of food (It's mostly at dinner) I get a severe stomachache. I also haven't been able to go to the bathroom well (it's just little pellets and I have to force it). I told my mother but she told me to take a laxative. We have the overnight ones. She told me that I have to have them over a week. Also I'm type 1 diabetic and she's been wrong before. She says I have a little cold and also told me to take pepto to take care of my stomachaches. Whenever I ask her to bring me I feel bad and my mom tells me the usual stuff.

Would I be wrong if I tell my mom to bring me to the doctor?
```

**Verdict:** KEEP

---

## 256 · `7eaa292609741895` · 610c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I bought a gun behind my moms back?

```text
My mom hates guns. She wants nothing do to with them. She doesn’t want me to buy I gun when I’m older, (not old enough to buy one yet), but it would make me feel safer in public. She has told me that I am being unreasonable and she goes around everyday without a gun and feels perfectly safe. That’s fine, but I’d like to have one just incase something happens. It’s better to have it and not need it, than to need it and not have it in my view. I feel if I do buy a gun when I get old enough my mom will resent me and it would cause tension between us.

Would I be wrong if I bought a gun behind my moms back?
```

**Verdict:** KEEP

---

## 257 · `7fdc66e65433a6a9` · 1541c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told my friends parents about her boyfriend they don’t know about?

```text
Just to clarify, I mean no ill intent towards anyone in this story, I’m just concerned about this girl. And if she somehow finds this, I’m sorry, I didn’t know what to do, and I’m worried about you. So this girl (now 18) met her now bf (21) online, and they’ve met in person maybe 4-5 times. He still lives with his mother, and according to people who have met him, he doesn’t appear to be a very good person. She hasn’t told her parents yet, and his behavior is concerning. She’s talked about how he pushes her boundaries when she’s clearly told him no, then gets defensive when I tell her that behavior like that isn’t good. He’s convinced her to go back to his (mothers) house and they were alone for a couple hours without anyone knowing where she was. He’s been making fun of her religion, and trying to convince her to become atheist. Regardless of the religion issue, I’m worried about the fact that he’s attempting to alter her worldview. He even wants her to move in with him when she graduates, and I think she’s going to, she’s said that she’s seriously considering it at least. She has actively lied to her parents about where she is and who she’s with to hide him. She isn’t a very mature person, and is attention seeking. She acts like she’s 10 when it comes to her mannerisms, and is easily manipulated. My mother has tried to subtly tell her parents, but they refuse to believe what is going on. And if IWBTA, what should I do instead?

Would I be wrong if I told my friends parents about her boyfriend they don’t know about?
```

**Verdict:** KEEP

---

## 258 · `80e472315ef06388` · 1145c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I stayed in my apartment till the end of the contract?

```text
So I am studying abroad and share an apartment with a friend. We signed a combined lease for the year, which officially ends on the 31st of May. My mum is flying over here around the 26th of May to help pack up, end the contract and take a short holiday before flying home. However, my roommate came to me today to say that she was planning on leaving on May 19th, which is the day after our university officially ends. She was hoping to pay a reduced rent for leaving early, but because im going to be staying longer she has to pay the full rent for the month of May. The landlord has said we can only pay a reduced rent if we get new tenants in straight away after we leave. She is now saying I should leave and go somewhere else so that we can save the money, i.e. put my things in the basement and sleep somewhere else. She said that I am forcing her pay more because of my plans. I just think its ridiculous go to all that trouble to save her 100 bucks, when the apartment is still equally mine until the 31st and I should be able to stay until I’m ready to leave.

Would I be wrong if I stayed in my apartment till the end of the contract?
```

**Verdict:** KEEP

---

## 259 · `81290354b76f4a8c` · 815c · ask=body+title

> **Title** (not part of the prompt): WIBTA for reporting Lyft/Uber drivers or giving them a lower rating for having a car that is obviously smoked in?

```text
I've had to take a lot of Ubers and Lyfts for work the past month, and more than half the cars have had extreme smell issues. It's either horrible BO (of the driver), extremely overpowering axe body spray or car freshener spray where I had to breathe through my mouth and could taste the spray, or most commonly, the car reeks of cigarettes. I'm an ex smoker and am generally very tolerant of smokers, I just despise the smell of old cigarettes that comes from smoking indoors (or in-car) because it makes me really nauseous. I know drivers rely on good reviews and giving them a negative review affects their livelihood, so I'm not sure if I should knock down their rating or report them.

Would I be wrong for reporting Lyft/Uber drivers or giving them a lower rating for having a car that is obviously smoked in?
```

**Verdict:** KEEP

---

## 260 · `81617f2f8e887664` · 1007c · ask=body+title

> **Title** (not part of the prompt): WIBTA for asking if chefs at a restaurant could clean off the grill before making my food?

```text
Hoping to get some answers from people who have some experience working behind the scenes at restaurants. 

I have to eat out with clients about twice a week and I do enjoy it but I often end up wasting my food. I’ve been a vegetarian since I was 11 for various reasons and I’m 30 now. It’s just not a thing I eat. Most of the time, any restaurant (in California) will have at least one veggie menu item so I don’t usually have a problem ordering anywhere. Anyways, I’ve noticed a lot of times, the cooks don’t clean off the grill that often, leaving rice grain sized pieces of burnt meat all over my food. Whenever it happens, I don’t know how to approach it without being a dick so I just don’t eat the food and say I’m full. 

I know it’s a super first world problem and I’m an asshole alone for wasting good food but is there any way to approach this before it happens without being an entitled putz?

Would I be wrong for asking if chefs at a restaurant could clean off the grill before making my food?
```

**Verdict:** KEEP

---

## 261 · `818264dd4ab1497c` · 1412c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I call out a kid for excessively aggressive gameplay on someone who doesn't consent to it when I play against him again in the future?

```text
It was last week when I was in an arcade gaming session with a friend of mine and two strangers, one of which is a kid as per title. The game was a racing game. Normally, all players in the same session must consent for brake blocking before the game starts. Brake blocking, unlike normal blocking, slows down opponent's car a lot and makes it harder to catch up. Such maneuver also drags out the game session, making it longer than normal. On top of this, such move is almost inescapable and render both players vulnerable to other players' cars catching up. But this kid proceed to engage in brake blocking against said friend without talking to anyone of us. I used my own defensive maneuver too which is part of normal gameplay, but I never brake blocked him. Said friend ended up lost against the kid while I got 3rd place behind the two of them with razor thin margins. Said friend stormed off to another cabinet. I thought he was pissed off on me, prompting me to ask why. It turned out that he was pissed off at the kid for the brake blocking gameplay. Thus, I talked to the kid's dad about how his gameplay is unacceptable. He seemed to understand my concern, but I doubt that the kid is willing to listen as he seemed to be unfazed by my remarks.

Would I be wrong if I call out a kid for excessively aggressive gameplay on someone who doesn't consent to it when I play against him again in the future?
```

**Verdict:** KEEP

---

## 262 · `831e31550e97e987` · 1124c · ask=title

> **Title** (not part of the prompt): WIBTA if I didn't share any of our dad's money with my sister when he passes?

```text
On mobile, forgive spelling and grammar please.

Bit of background: My father is not a particularly good person. In fact if I were pressed I'd be forced to admit he's actually quite a bad one in many respects. His behaviors led to the termination of his relationship with my sister nearly fourteen years ago. I don't want to go into it, but she hasn't spoken to him since, and her reasons for doing so are good and I respect them.

Recently, he has been making preparations for his assets for his retirement and estate. As I still maintain a relationship with him, he has told me that I will be the listed beneficiary to essentially everything when he goes, and to "figure it out" with my sister when that happens.

My sister wants nothing to do with him anymore, and I'm virtually certain she will refuse anything offered. Would it be ethical to just keep everything if she doesn't accept it? At the moment it looks like when our parents pass we'll be splitting whatever our mother leaves us, but I'll be getting everything from him.

Would I be wrong if I didn't share any of our dad's money with my sister when he passes?
```

**Verdict:** KEEP

---

## 263 · `834e545f46080a4d` · 726c · ask=body+title

> **Title** (not part of the prompt): WIBTA for wanting to defend myself?

```text
A few months ago a friend and I had a falling out. It's been over 6 months since I've seen them but I still feel bad about what happened. I sent them a text the other day just for some closure and to apologize. They responded a weekish later being incredibly hurtful and accusatory.  They didn't accept my apology or apologize themselves at all. On one hand I want to take the high road for myself and also not get more texts belittling and bashing me. But part of wants to text back and defend myself because the things they said weren't true and were just hurtful for the sake of being hurtful.  Would I be an asshole if I texted back in my own defense? Or should I let it go?

Would I be wrong for wanting to defend myself?
```

**Verdict:** KEEP

---

## 264 · `841264151d56ba38` · 1025c · ask=body+title

> **Title** (not part of the prompt): WIBTA for canceling an interview on the company’s dime?

```text
I’ve been interviewing with a company and I have to fly out to their headquarters to interview. They booked me on a flight that has one connection which is a little annoying but not a problem. The problem is my flight out of my home city is delayed and I won’t be able to make the connection. I contacted the company and they said to go to the airport anyway and talk to the gate agent. I talked to the airline and they said there are no flights out of the second airport that day. The company is telling me to take the flight anyway and try to make the connection with the risk of me being stranded. There is another flight out of my home airport that leaves at 5am and gets me there in certain time but this situation has caused a lot of inconvenience with me having to take time to go to the airport on my own dime coupled with the risk of being stranded in another city that I am contemplating just canceling or asking to reschedule. Am I being a bitch?

Would I be wrong for canceling an interview on the company’s dime?
```

**Verdict:** KEEP

---

## 265 · `843f1b3885dc20f1` · 678c · ask=title

> **Title** (not part of the prompt): WIBTA for ditching my own birthday party

```text
tonight my friends are throwing me a birthday party in one of their apartments. 

the issue is, that it’s not my party. they all invited a ton of people i️ don’t know, and it’s a lot bigger than i️ ever wanted (50+ people)

it doesn’t really feel like my party and it wasn’t what i️ had agreed to when i️ originally said i’d go
i️ don’t feel like going and babysitting a ton of drunk college students

however my roommate is really excited. she bought decorations and snacks, and a tiara for me for the party. i️ don’t want to be an asshole to her and dip, but i️ really don’t want to go to a rager and pretend it’s my party

Would I be wrong for ditching my own birthday party?
```

**Verdict:** KEEP

---

## 266 · `847d1fa8ab006992` · 693c · ask=title

> **Title** (not part of the prompt): WIBTA if I showed up with all of my soon to be ex’s stuff at her dorm room to break up with her?

```text
I’m going to end my current relationship.  I live in a house and she has a dorm room, but stays with me almost every night.  She has brought quite a lot of her stuff over here.   She really only uses her dorm room during the day in between classes.  

She has a real temper and bad habit of breaking things/getting physical when she’s angry.  I’d really rather not break up with her at my house because it’s extremely likely that stuff would get damaged.  I’d like to just pack all her stuff up tomorrow and wait until she’s at her dorm room in between classes and break it off there.

Would I be wrong if I showed up with all of my soon to be ex’s stuff at her dorm room to break up with her?
```

**Verdict:** KEEP

---

## 267 · `84912ed793b2844d` · 901c · ask=title

> **Title** (not part of the prompt): WIBTA if I confront this dog's owner?

```text
I was on a walk with my dog and get to the end of the street, which is about 1 mile from my house. We get to basically the end of the road and a dog CHARGES from its driveway into the street, at my dog, and at this point they are both growling and barking at each other.

My dog got scared and escaped from her collar, both dogs still bucking at each other until the other starts chasing mine back toward the direction of my home but stopped about 30 feet after. I chase after my dog to make sure she goes back home and never booked it so hard in my life

Not only was I concerned that my dog was going to bite the other, but there would have been hell to pay if my dog got hurt lol.

Animals may be territorial but this one coming away from its home and into a public street to go after mine is not okay, and I'm thinking about confronting the owner.

Would I be wrong if I confront this dog's owner?
```

**Verdict:** KEEP

---

## 268 · `8494ced2ef63da5c` · 876c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I said no to celebrating my nephews birthday at my wedding?

```text
My soon to be sister in law has asked me and my H2B if during our reception party, we can pause everything to sing Happy Birthday to her son who will be 6. Firstly, she doesn’t really make an effort with her sons birthdays any other time apart from getting the family round for a meal, so it strikes me as the only reason she is wanting this fuss is because she knows there is essentially a free party for him which happens to be our wedding! However I do see her reasoning in that our wedding happened to be on his birthday (we completely forgot this when booking the wedding) and it does need to be recognised one way or another I just really don’t feel comfortable with pausing everything for one person, I only plan on getting married once whereas this kid will have lots of other birthdays.

Would I be wrong if I said no to celebrating my nephews birthday at my wedding?
```

**Verdict:** KEEP

---

## 269 · `84d6c96fcc804dc4` · 487c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I ask my roommate to stop buying single-ply toilet paper?

```text
This is the dumbest issue and part of me is ready to just drop it and deal with flimsy TP. But I hate using single ply, and I hate making guests use it even more. Even my shitty job doesn't have single-ply in the bathrooms.

I bought some nicer stuff recently, but when that ran out he got more single ply. I should probably just always be the one to buy the toilet paper if I have a problem with his, right?

Would I be wrong if I ask my roommate to stop buying single-ply toilet paper?
```

**Verdict:** KEEP

---

## 270 · `85c019de2e650036` · 1294c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for talking to my boss about my current time off situation?

```text
Okay so, I was made part of middle management the beginning of 2018 at my job, and at the time there were no rules about time off. I was told I would be given five paid days off, which I thought was amazing but was never told anything about unpaid days. After a long year I took nine days off, five of which were paid, in the end of December/start of January. I had requested said time off in October as I planned on taking a small trip on my time off. In the beginning of February I was told that the time off situation had been changed because of an employee taking off a bunch of days. I was now told that as middle management I was still allotted my five days paid, but now was only allotted five unpaid days. And that my time in the beginning of the year has been retroactively counted against me. I’ve been pretty salty about this whole thing since it happened, like I’ve complained to a ton of people because it seems unfair to change policy after the fact. And I’m mad now because I try to take two small vacations a year, which I can still do but if something else comes up like a wedding or concert or something I’m screwed. I should also add in, everyone else is allowed to take as much time as they want.

Would I be wrong for talking to my boss about my current time off situation?
```

**Verdict:** KEEP

---

## 271 · `85e2edd6d29d44f4` · 1166c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I cut off my best friend on my birthday.

```text
So tonight is my birthday "party". I have only 2 friends and we were going out for drinks. 
My best friend just told me that she can't go out because she doesn't have any money. 
She lives off the money her dad gives her and he gave her like $300 on monday. She paid her phone and yesterday went shopping and blew out the rest on clothes. She sent me photos of everything she bought and even ordered more online. 
So now she tells me she can't come because of money even when SHE planned my party and she knew I had been having a hard time with college/family and I was really looking forward to relax and celebrate one night. 

We have been friends forever and have had some ups and downs but I'm really mad and I really feel like someone who cares about you would never do that, knowing I love my birthdays. (we even offered to pay if she comes but she "Would feel baaad" )

Am I over reacting or she really doesn't give one shit about my feelings? 
I haven't text her back because I don't know if I should brush it off and suck my anger or go nuts on her and tell her how I really feel. 
Please help.

Would I be wrong if I cut off my best friend on my birthday.?
```

**Verdict:** KEEP

---

## 272 · `85e5ed2f76ee4160` · 716c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if i really shout at a manager of a repair shop

```text
So I had bought headphones online with one year warranty it was pretty good till it stopped working around 3 months in around October I gave it to repair since the company outsourced the repair to another company the manager was of that shop said me to take the headphones home and that the company which I purchased from will replace it in a week. So 2 months pass by I get pissed off I call the company which I purchased from they said that they don't replace the headphones and that according to their system I never entered the repair shop so they added the repair shop person on the line. Now he started to say me to bring it to the shop again.

Would I be wrong if i really shout at a manager of a repair shop?
```

**Verdict:** KEEP

---

## 273 · `867f56dd1da894df` · 943c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I ask my friend to pay half after I cracked his phone?

```text
Sorry for formatting I'm on mobile. Context, I was in school and we were in the yard and I was at the other corner of the yard going with my friend to the bathroom, when I was coming back to my friend who's phone I broke i was jogging towards him as he was recording me. So after jogging towards him and he's still recording I purposefully collide with him (he was standing still recording the whole time) and his phone slips out of his hand, falls and cracks. My friend asked me to fix the phone and I agreed to do so but thing is, I didn't crack his screen protector, i cracked his phone's glass. I asked him where his screen protector was and he said he "didn't have one cause he never needed it" I thought this was stupid because screen protectors are in case you need them not something needed straight after you buy them. (He has an iPhone 6s with no applecare)

Would I be wrong if I ask my friend to pay half after I cracked his phone?
```

**Verdict:** KEEP

---

## 274 · `87664f459ca0216d` · 534c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for telling my landlord I intend to break my lease because of mice?

```text
I moved into an apartment last October and since then have seen and heard mice. I’ve reached out to the landlord numerous times and he’s just told me to buy mouse traps. Beyond that he has not tried to remedy the situation. I asked about him calling an exterminator and he said I would have to pay for them to come out. I want to tough it out until my lease ends, but I keep finding droppings on my kitchen counters and I‘m honestly at my wits end.

Would I be wrong for telling my landlord I intend to break my lease because of mice?
```

**Verdict:** KEEP

---

## 275 · `878cf601f9018a68` · 2052c · ask=title

> **Title** (not part of the prompt): WIBTA If I took money from friends?

```text
Okay, so before you go off the title for my judgement, there’s some details I need to explain first. 

So my best friend from home and I have been talking to each other lately and He knows that in my current situation, it’s very difficult for me afford anything right now. I’m currently attending college, have no job or meal plan, so food can be very scarce. He’s even come and visited me seeing how much I struggle. However, every once in a while, he’ll offer to send me money (about 40), just for an emergency. But I’m always declining because 1. I don’t want to owe him any money, and 2. As much as a good friend he really is, I don’t want to take his money because he has his own stuff to pay and deal with. So I don’t want to be a burden to him. I’ve tried to apply for jobs. But unfortunately no luck as no one wants to work with my school schedule. I have to save the money I currently have however to pay for my rent. 

Now recently I’ve been able to budget, but my rent is due soon so I’ve been going extreme lengths trying not to spend money. I bought a bunch of groceries to last for a month, but believe or not, my roommate at everything with the span of a week. He said he would try and get SOME groceries, but he doesn’t get paid until next Friday. And I ate my last batch of food this morning for breakfast. Now I can probably hold off till the weekend, but it’s going to be very difficult to do considering it’s barely the middle of the week. The best I can do is wait till my roommate gets paid, or at the very least wait till my parents come visit me next weekend and for them to restock groceries for me, as they always do whenever they come visit. 

It’s getting really difficult to eat one meal - two meals a day, and honestly I’m kinda thinking about asking him considering this is looking like a definite emergency. If I don’t get groceries soon, I’ll basically be doing a water fast until I get more food. But at this point, I rather have a roof over my head then be able to eat.

Would I be wrong if I took money from friends?
```

**Verdict:** KEEP

---

## 276 · `879be3aeb4d0890a` · 663c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for leaving a note on a shared bag of cookies?

```text
My stepdad sometimes buys everyone a small bag of cookies from the supermarket and leaves them on the kitchen counter. They come in a little brown resealable bag. It’s important to reseal the bag, because otherwise the cookies get stale. The problem is that I often find the bag open. I know it’s not me, since I’ve been making sure that I close the bag. The only other person who eats them is my brother, so he must be the culprit. I was thinking of leaving a little note on the bag telling everyone to close it, so if anyone takes some, they will close it. My mom says this is passive aggressive.

Would I be wrong for leaving a note on a shared bag of cookies?
```

**Verdict:** KEEP

---

## 277 · `87ff6f3923769d81` · 1028c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I didn't show up to a wedding because my name isn't explicitly on the invitation?

```text
Boyfriend: A;  Groom: B; Bride: C Me: me/I A, B, and I all went to the same school and the three of us get along fabulously (I'm a girl who naturally gets along better with men, though I always reach out to women to the same extent when socializing). A is going to be a groomsman at B&C's wedding this summer. C is a friendly and outgoing gal, though her relationship with me is cordial at best. We always knew they were going to get married, so in the back of my mind I was of course excited to go to their wedding. So imagine my surprise when only A's name showed up on the save the date, invitation, and RSVP card - all handwritten in that Pinterest-y calligraphy font (you know the one). Now A and I have been together for longer than B&C, so it's not like it was a question whether or not we would have gone together...it must not have been a mistake if three separate pieces of mail excluded me. A is mum on the matter.

Would I be wrong if I didn't show up to a wedding because my name isn't explicitly on the invitation?
```

**Verdict:** KEEP

---

## 278 · `8830ae52cc646e66` · 604c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I refused to take orders at the drive thru window?

```text
I work at a fast food restaurant. We only have one window and only one person works that window for their shift. This means I am taking orders, taking money, giving it back, bagging food, utensils, and drinks, and handing out food during rushes and during slow times. I am also supposed to support the front of house when possible. So many times people will order their food, pay, wait until they get their food, and then add on a sandwich or something. This ruins my drive thru times and the whole system I have going when working.

Would I be wrong if I refused to take orders at the drive thru window?
```

**Verdict:** KEEP

---

## 279 · `89d3ad7bb385cca6` · 1187c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I exposed my roommate and his gf

```text
Context: I live in a pseudo-campus, like regular campus but more services and more rules to follow. For example girls and boys cant sleep together (or you'll get kicked) So there is my roommate, an ok person BUT I cant communicate with him in any way. I have just been asking him for the past months to keep his sh%% clean. He won't wash the dishes until he run out of them, won't dump the garbage until theres no room left and all the shit roommate thing. But thats not all 3 months ago he got his new gf, who his as stinky and naughty as him, and they directly started living together. They produce loads of trash that sits there for weeks, and there's a ton of trash everywhere everytime. I asked them more than once to PLEASE take at least the trash out every other day but ignored. I'm really pissed because I always keep all my stuff in place and I also have people that comes over. I didnt want to come to this but since 3 4 days ago there is this shit stink of fish everywhere in the house, and that's not even bothering them. Since im really fed up, I will just expose them as soon as I get home and eradicate this problem.

Would I be wrong if I exposed my roommate and his gf?
```

**Verdict:** KEEP

---

## 280 · `8a1ffbbd60389dfd` · 1071c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I ask for a discount on my scooter?

```text
So, on February the 15th I bought a new scooter. They didn't have it at the shop, so they had to order it from the supplier and then I could pick it up the next week. I said that it was no problem, I know the type I bought already, so I paid them 1700€. Few days go by and I get a phone call that the supplier didn't have the colour I wanted (black). And if I wanted that colour, that it would take a month before it arrived from the manufacturer. I was a bit disappointed, but really wanted a black one so I agreed. 5 weeks later i went to the dealership and asked for an update about my scooter. "Yeah, something went wrong at the supplier and it will take another month before we can order it." Wich means I will have it by the end of April, beginning of May. So this time I'm very disappointed and a bit sad because now I have to go another month with public transportation to my work, which cost me 5 times more than it would by scooter. But because I'm not great at confrontations I just said "oke" and left.

Would I be wrong if I ask for a discount on my scooter?
```

**Verdict:** KEEP

---

## 281 · `8a4f2d54aa824f09` · 621c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I started ghosting her?

```text
Alright so there’s thing some guys tend to do when they get rejected, and that’s just stop talking to the girl altogether. I recently confessed to a girl and she politely declined and I understood, we both agreed to stay as friends. I’ve been trying to start conversation or text her in an effort to be a friend like she agreed, but she always just dry texts me back. Stuff like, “lol yeah” or “thanks lol” she’s been making it crazy difficult to “be a friend” like she agreed and I don’t want to come off as one of those dudes who stop talking to her because she rejected me.

Would I be wrong if I started ghosting her?
```

**Verdict:** KEEP

---

## 282 · `8b2167d97fee9dd6` · 610c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I submit a piece of work just on my behalf instead of both mine and my partners?

```text
So long story short, I  got lumped with an idiot in my music class, and he literally never tries. For our work, we had to choose an audio file, but while I wanted to do one, he wanted to do a different one. He turned off the computer in anger because of this. He also did absolutely nothing we were supposed to; he listened to raps with his friends while I sat there working on the computer. As a result, our piece of work isn’t nearly as good as I had hoped it would be, and I desperately want a good grade.

Would I be wrong if I submit a piece of work just on my behalf instead of both mine and my partners?
```

**Verdict:** KEEP

---

## 283 · `8b241bccdd19a9a9` · 1254c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA For not asking for a day off so my family hosting Christmas can do it at a more convenient date?

```text
Before I get into everything I want to say I WANT THEM TO JUST HAVE CHRISTMAS WITHOUT ME!!!!!!!! So my family members hosting Christmas for the family don’t want to do it on Christmas Day because they work the next day and the set up and cleanup is tiring and they want a day to relax. Ok, 100% understandable that they want to do it earlier when they have the next day off. However since I’m the youngest in the family (I’m 20) they are hinging the date on whether or not I’ll be able to go on the earlier date. Any other year I would take the day off, however I’m already taking 8 unpaid days off before the new date because I’m visiting my dad in another state. I’m also getting shit hours for some reason and have barely any money so those 8 days unpaid is gonna hurt and I really can’t do a 9th. So I just want them to do Christmas on the date that’s convenient for them, and I’ll just miss out this year. My grandma is pressuring me to take the day off saying I’m being selfish. Also worth noting, 7 days of the 8 I asked for and got months ago, the 8th was more recent and they kind of had to give me because it was college related.

Would I be wrong for not asking for a day off so my family hosting Christmas can do it at a more convenient date?
```

**Verdict:** KEEP

---

## 284 · `8c5d75d9519b40db` · 952c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I asked my neighbor to turn down their music?

```text
Their music is *blaring*. I can clearly hear- and feel- the bass shaking whatever surface the stereo is on, the melody of the songs, the instrumentals, and almost fully make out the words. I'm half jealous of their sound system tbh. 

It's almost 11am on a Saturday. They've been playing it since about 10 am. I work nights and I need to get to sleep. My schedule is completely flipped by this point, and I'd usually be in deep sleep right now (I think), but I actually had business to take care of. I generally cut my neighbors slack when their kids make noise because... well, they're kids. I doubt the kids are home now, and this might be the parents' "downtime." But this is a lot right now. My thing is, it's 11 am on a Saturday. If anytime would be okay to play music, it'd be about this time- right? Would I be an asshole for asking them to turn it to like, half what it is now?

Would I be wrong if I asked my neighbor to turn down their music?
```

**Verdict:** KEEP

---

## 285 · `8cd1cba47d668d20` · 485c · ask=body+title

> **Title** (not part of the prompt): WIBTA if i told my roommate he needs to get back to the dorm earlier if he wants to smash?

```text
He's bringing some girl over Friday night and he wants to smash, which is fine, ive got plans friday too, but he said they aren't gonna be back til like 1:30, and they want the room for 45 minutes. Im sorry, but ive got plans, but i dont want to sit around doing fuck all until 2:30 in the fucking morning. Would i be a dick to say, if you wanna smash at the dorm, get back earlier?

Would I be wrong if i told my roommate he needs to get back to the dorm earlier if he wants to smash?
```

**Verdict:** KEEP

---

## 286 · `8e0f55631647250d` · 480c · ask=body+title

> **Title** (not part of the prompt): WIBTA For feeling i don't owe my mom anything?

```text
Now, this hasn't happened. I'm using this for future reference.

My mom and I are close, and she's been supporting me all my life. Even though she has done this for me, I don't feel like I owe her any money or any objects (like jewelry) Because she raised me. It was her choice to raise me and invest in me, I didn't ask to be born. Of course I'd support her but Am I an asshole for feeling like I don't owe her anything?

Would I be wrong for feeling i don't owe my mom anything?
```

**Verdict:** KEEP

---

## 287 · `8e382420e510aff1` · 825c · ask=closer

> **Title** (not part of the prompt): WIBTA Pizza dilemma

```text
Context: I live with five other roommates. The apartment we live in provides one fridge. As you can imagine, the fridge gets pretty full with six people sharing it. Luckily we’re all pretty good about sharing the space with each other. However, whenever “Cinderella” orders a pizza instead of wrapping the leftovers in tinfoil or plastic wrap she haphazardly shoves the entire box on top of everyone else’s food. This makes it hard to get food or open the fridge without the box falling out.

Next time “Cinderella” orders pizza I was thinking of saying something like “ If it’s not too much trouble would you mind wrapping up any leftovers in tinfoil or plastic wrap. The fridge space is limited and I think this would help create more space ...”

Does this come off as rude or micromanaging? Let me know...

Help me decide.
```

**Verdict:** KEEP

---

## 288 · `8e4b9f51dc6aae1d` · 735c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for doing a separate, more thoughtful retirement gift?

```text
One of my department coworkers is set to retire at the end of March and people are starting to plan gifts, party, etc. One of the other department members sent out an email with a generic, "funny retirement" gift and suggested we all go in on it for coworker. Everyone quickly agreed and that was the end of it. Normally this type of gift would probably be fine but the person retiring is a very thoughtful and caring coworker. I've talked to them a bunch regarding their post-retirement plans and came up with a really nice, sentimental gift idea that compliments their plans nicely. This gift would be more expensive than what the department is planning to do.

Would I be wrong for doing a separate, more thoughtful retirement gift?
```

**Verdict:** KEEP

---

## 289 · `8e5f052cf4f70e5f` · 813c · ask=body+title

> **Title** (not part of the prompt): WIBTA for denying my friend rides home from work an hour before I get off?

```text
Right now I'm looking for a place to stay and I'm currently living with a group of friends in their apartment. I happen to work with 2 of them. The apartment is less than 5 minutes from where we all work so my friend wants me to give him a ride. No problem I thought until he got off an hour early and asked if I could drive him home then. He already asked our boss who said it was fine but I would have to clock off and clock back in when I returned. So I've done this twice now and honestly it's kind of annoying ends up taking 15 or so minutes each time and I just come back into work just to make the same ride 45 minutes later.  Just want to know would it be rude to deny him a ride since I'm staying with them rent free?

Would I be wrong for denying my friend rides home from work an hour before I get off?
```

**Verdict:** KEEP

---

## 290 · `8e81e39a161dc3d8` · 1168c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I slept with a woman my brother is interested in, even if she doesn't reciprocate those feelings ?

```text
Please read fully before answering  So there's this woman we both know, she's actually a co worker. My brother visits work quite often so she got to know her. He got her number and they've been talking. She's quite beautiful and I was also interested in her but I'm too shy. We've had conversations but I wouldn't say we're friends or anything like that. Anyway, it turns out she just wants to be friends with him, or atleast that's what my brother told me. He said he tried but it was obvious she wasn't interested. Oh well. A few days later I find out by another co worker that she's actually into me (she even told her that she was frustrated that my brother talked to her instead of me) which is a huuuge surprise. I'm not too popular with the ladies if my username didn't make that obvious so I really want to jump at this chance but my brother isn't just a family member, he's one of my best friends and I wouldn't want to hurt him. He made it pretty clear that nothing was gonna happen between them but I feel like maybe that's because of me.

Would I be wrong if I slept with a woman my brother is interested in, even if she doesn't reciprocate those feelings?
```

**Verdict:** KEEP

---

## 291 · `8f65978859ba4d17` · 1804c · ask=title

> **Title** (not part of the prompt): WIBTA For keeping a bonus to myself?

```text
A little back story about those friend and me vs bonuses:
We usually do the same job with three or more, in which one joins and he refers the others, because those jobs usually give a bonus varying between 50 and 200 bucks. I've always been fair at the deals we made, where the first person refers the others and splits each bonus with the person he got the bonus for. Which always means the first at the company needs to learn how the job is done, explains it to the others and gets 50% of the total amount for explaining and managing it all.

So a few years ago, I would refer 6 people, which would grant me 150$ per person, of which each of them 6 would get 50$, which basically means everyone would end up with 50 and I would end up with 600. After I referred the first(I'll call him Bob for now), I then explained the job to everyone. And then Bob referred the others behind my back, though I did the work for it. So he basically made 550 off me.

They basically broke the deal and soon after we didn't talk much anymore. Now with other friends I'm still having the same compromise.

I would explain everything to em and then refer the both of em. Each one would grant me 100 bucks, I then would split each bonus in half and keep 100 bucks in total myself.
Now the second guy, just told me he's gonna grt referred by the first guy I referred myself, after I explained everything. Normally I wouldn't make a big deal out of it. But the reason why he did this, is "because the first guy needs it for a big holiday", while I'm just working to finance my study.

TL:DR

I made a deal of referring two people, granting each of em 50 bucks and myself 100 bucks, then behind my they changed so one of them gets more. I now feel like keeping my bonus myself.

Would I be wrong for keeping a bonus to myself?
```

**Verdict:** KEEP

---

## 292 · `8f71418ffd75cdc2` · 1469c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I uninvited my friend to our party?

```text
I’ve been friends with this person off and on for years. When we get a long it’s great. When we don’t it’s awful. Over time the bad started building up. He ditched me at party that he brought me to as a plus one to get laid (and will occasionally just get up and leave in the middle of hanging out bc he planned something else without mentioning). Then that same night asked me to come pick him up bc I guess things weren’t going well. Complained when I said I was drunk and couldn’t be of any help. He continues to be not emotionally supportive or understanding. When I was having a bad day & we were supposed to hang out, he opted to cancel if I was gonna “be like that” as in not being in a good mood. Anytime I would tell him things I didn’t appreciate (for example flaking on plans with no texts or calls to cancel) he just threatens to bring up past things I have done wrong in the friendship. Did not even attempt to say bye to me when I was leaving across the world for 5 weeks. I’ve had a hard time feeling justified in ending this friendship for many reasons, but I’ve opted to just not reach out anymore. Me and my roommates are having a party and one of them invited him (not knowing I’m not talking to him). I honestly doubt he would come, but I don’t want to feel backed into a corner if he does decide to come. I feel like many of the reasons I have are small & maybe I’m just being over dramatic?

Would I be wrong if I uninvited my friend to our party?
```

**Verdict:** KEEP

---

## 293 · `8faeecfcc5581566` · 408c · ask=body+title

> **Title** (not part of the prompt): WIBTA for selling the gun he bought me?

```text
TLDR ex bought me a gun after he quit his job, and I don't want it should I sell it?

   My former boyfriend bought me a gun, and I don't want it, I'm not going to use it, and we have two kids on the house, and no safe. He said he dosen' t want it I need money for well everything, he told me he would be upset if I sold it, but I don't want it in my house

Would I be wrong for selling the gun he bought me?
```

**Verdict:** KEEP

---

## 294 · `90218f5b07ccbfdd` · 701c · ask=title

> **Title** (not part of the prompt): WIBTA if I ruin the company I work, by quiting my job

```text
So I have a predicament, I work in a small business (2 owners and in total 3 employees including me). I have been working here for about 8 years now.

And I recently found out that I can make more money and have double the time of I have now.
Easy decision,,except that they have trouble finding new employees and they need to have a certain amount of people by law. And both my other colleagues are leaving.

Would I be a asshole if I also quit and possibly ruin their chance to stay in business.

On a side note to make it more difficult, we work in the transport sector and my father own the company that they get their cargo from.

Would I be wrong if I ruin the company I work, by quiting my job?
```

**Verdict:** KEEP

---

## 295 · `905b7e22d3a6c378` · 437c · ask=title

> **Title** (not part of the prompt): WIBTA if I told my girlfriends manager that she spat in someones drink

```text
A few nights ago a customer came in that her co-worker really didn't like that did some bad stuff to her and ordered a coffee and they both decided it would be okay to spit in her coffee. I'm honestly disgusted and I'm shocked that she would do that. I think it's completely not okay to do that but I was wondering if it was petty if I told her manager.

Would I be wrong if I told my girlfriends manager that she spat in someones drink?
```

**Verdict:** KEEP

---

## 296 · `90f0cce26473c109` · 1076c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I found a new group of friends?

```text
I have been friends with a group of people since the beginning of college (as of last August) and recently have been itching to find new friends. My only concern is that I have become pretty close to a few of the people and would feel incredibly guilty for potentially doing this, but the feelings of annoyance and reluctance to hang out have been overpowering recently. It's hard to explain but we just aren't clicking right and I have kind of dug myself into a hole by faking my feelings the past month or so. The reason I am having a hard time is because one person in particular really became attached to me, which is fine, but this person does not have a lot of other friends and I would feel like a jerk if I ended up removing myself. This person has a lot going on in their personal life and I feel bad but im just not sure how much longer I can bring myself to "fake" being friendly when I can't help but get annoyed whenever I am around them. Has anyone else experience anything like this or am I being an asshole?

Would I be wrong if I found a new group of friends?
```

**Verdict:** KEEP

---

## 297 · `91300f8bd0dd1271` · 612c · ask=title

> **Title** (not part of the prompt): WIBTA if I went to a dealership to test drive with no intention of buying?

```text
Last week I went to my local Ford dealership to get a part for my mustang, and immediately as I walk in I’m greeted by a salesman and he asks if I would be interested in trading my car in. I told him I’ve though about it a couple times for in the future but haven’t really considered it. He then offers me to come back another day to test drive some newer mustangs. 
           
           I am in no way financially set to afford a newer mustang and I don’t want to go in there and waste his time because I know I won’t buy.

Would I be wrong if I went to a dealership to test drive with no intention of buying?
```

**Verdict:** KEEP

---

## 298 · `916e255c6f4846cf` · 396c · ask=title

> **Title** (not part of the prompt): WIBTA If I say I’ll pay my friend for something but don’t?

```text
My friend owes me $15, he agreed to buy me wine as payment months ago. I’ve asked him quite a few times to get it for me but he always has an excuse. I’m thinking if I say I’ll pay him for it he wouldn’t have a problem, but I don’t think I should pay him because he owes me it- yet he won’t buy it for me otherwise (thus far)

Would I be wrong if I say I’ll pay my friend for something but don’t?
```

**Verdict:** KEEP

---

## 299 · `92786f1a553c68a1` · 1388c · ask=body+title

> **Title** (not part of the prompt): WIBTA If I told my sister not to move back to CA?

```text
My sister and her husband moved to SC from CA in 2012ish. From what we know, he left CA to get away from his family and my sister wanted to come back home. Our mom was gracious enough to allow them to live in one of her properties for less than $800/month and has "borrowed" at least $5000 from our mom. They have always been tight and irresponsible with money. They have a 9 month old baby and when they have to work night shifts, my brother or mom is always available to watch him.

My sister is an EMS tech and he works at a gas station. Neither have degrees and both forget that the cost of living in CA is different than in SC. The help they have now will be gone if they decide to move and/ or come back. His parents have somehow convinced him that because paychecks are higher on the west coast, they will be okay but forget that the COL is higher as well. They struggle to pay rent and to keep up with what they have now.

We understand that her husband's family wants to be close to their son and grandson but they have been looking to live about an hour away from his parents. 

Originally, my sister said he can move back and she'll just see their son in the summer but all of a sudden she is okay with going

Does that make me/ us the asshole or just selfish for not wanting them to make an irresponsible decision?

Would I be wrong if I told my sister not to move back to CA?
```

**Verdict:** KEEP

---

## 300 · `92b242af3cef88f3` · 1504c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I refused to play translator for a holiday I'm taking with friends?

```text
I'm planning a trip with 3 or so friends to go to France in the summer. Now I speak fluent French, although I'm rusty, having done it for 7 years through school (ages 11-18) and using it during my degree for research. 2 of my other friends also speak the language to varying degrees, mostly at a conversational level. Now something has cropped up which is bothering me- namely that on a few occasions I've had members of the group openly assume that I'm going to be a translator. On one occasion we were looking up some tourist sites online and some art thing we found was in French only. One of the friends who wanted to go to this was saying "well that's not a problem, you can translate", making the assumption that I'd be happy doing this, which I'm not. I said this at the time and it caused something of an argument, with her saying I'm being selfish and me saying that it's more so to simply expect that I'm going to do something without asking. As far as I'm concerned, it's my holiday too and the last thing I want is to be a workhorse, especially when in at least two people's cases they're perfectly capable of working things out for themselves using a phrasebook or simply learning stuff beforehand. Neither is it totally impossible if they're that worried about their language skills, especially with the Internet, that we couldn't find things to do which have the option of using the English language.

Would I be wrong if I refused to play translator for a holiday I'm taking with friends?
```

**Verdict:** KEEP

---

## 301 · `9330c0e2ee0b0b5a` · 1531c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if i call my brother’s girlfriend out for her behaviour?

```text
My brother and I has always been close, even with his multiple girlfriends in the past. We share multiple hobbies like gaming together and we have a great relationship. About 6 months back, he got a new girlfriend and ever since then, I have been spending lesser and lesser time with him due to his girlfriend’s obvious displeasure with me spending time with him. Do note that I have not done anything that offended her, she’s just really attached to my brother. So, everytime I start chatting with my brother, it never last more than 5 minutes because she would just be sitting beside quietly and showing a face of displeasure. When she’s at work and my brother gets to game with me, she expresses displeasure when my brother doesn’t reply her in time. Due to this, we hardly game now as she limits and complains about his gaming time. (Which is usually only 2-3 rounds?) Look, I understand that couples have to spend time with each other and all, but they are literally stuck at the hip 24/7. She’s always hanging around my house and has been really incosiderate to my family but laughing and talking really loud even at night. (She has a really high pitched voice.) There has been instances where she was rude to my parents as well. One instance was that she walked out of the room in anger when my mom said something that opposed her views. Usually, I am quite a amicable person and don’t like to pick up fights, but this is really getting out of hand.

Would I be wrong if i call my brother’s girlfriend out for her behaviour?
```

**Verdict:** KEEP

---

## 302 · `936e07872cf91a1e` · 1516c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I end it with my personal trainer when I still have 4 paid for sessions left?

```text
Last fall when I moved to a new city, I joined a new gym and decided to get a personal trainer. Our initial session went really well, so I decided to take the leap and buy her 10x package (yes yes, my mistake). Fast forward a few months, I have 4 sessions left (I was sick for like a month and there’s been holidays in between) and I just don’t want to do it anymore. I’ve found a good gym routine on my own, plus taken up a sport so I get a good amount of exercise in my free time, and simply put I just don’t really enjoy her training style. The first few times were nice because I learned new things etc, but at this point I never feel like going, and I kind of feel like I’m wasting both of our times. The only issue is I still have those 4 paid for sessions, so I’m worried it would be a slap in the face to her to virtually be like “I’d rather throw this money away than train with you”. Furthermore, I genuinely have to cancel/move next week’s session due to something coming up, so I’m worried that would be like a double whammy. She’s really nice and it’s nothing against her as a person, it’s just not working for me. TL;DR: Bought a 10 session packet with my personal trainer. I’m not enjoying it and don’t feel like I’m getting much out of it, so I want to stop going, but I have 4 paid for sessions left and am worried how it would come off to stop seeing her when she knows it means throwing money away.

Would I be wrong if I end it with my personal trainer when I still have 4 paid for sessions left?
```

**Verdict:** KEEP

---

## 303 · `95325a915ed168b4` · 2066c · ask=title

> **Title** (not part of the prompt): WIBTA if I told someone they should really use headphones in the gym?

```text
So the gym in my apartment complex is pretty small. I guess you could say it’s a “semi private” gym because it’s only open to residents of the complex. It’s one room and has about 4 cardio machines,  and 1 weightlifting machine. The entire room I can estimate to be about 25 ft long and 25 ft wide, so being such a small space naturally it can get a little cramped at times. 

There’s this girl who started working out at the start of January. The first time she came in, she started playing her music out loud. My first reaction was to give the benefit of the doubt. It’s a new year. Maybe she’s just starting out and hasn’t gotten around to buying headphones? Okay that’s fine. But it’s been weeks now and every single time she comes in to workout, she plays her music out loud for everyone to hear. Even with my headphones on and the volume turned up, I can still hear the drone of her music in the background. I know other people in the gym are annoyed, or seem to be cause I’ve seen a few people giving some stares, but no one has ever said anything. 

Maybe I’m being way too sensitive and making this up, but isn’t it customary to use headphones when listening to music while working out in a gym? Especially if there’s other people there? Would this annoy anyone else or am I being dramatic? I just don’t understand cause personally, I would never in a million years even think about forcing other people to listen to my music out loud in the gym. I’d be too self conscious and worried that I was annoying other people! 

I’ve thought about saying something to her, politely of course. But I haven’t because 1) I hate confrontation and 2) I’m still unsure if I’d be an ass for even saying anything. 

I understand the gym doesn’t revolve around me, and if it bothers me I guess that’s my problem. But at the same time I think it’s quite rude of her to constantly play her music out loud because every single gym I’ve ever worked out in that’s always seen as bad gym etiquette.

Would I be wrong if I told someone they should really use headphones in the gym?
```

**Verdict:** KEEP

---

## 304 · `95d7bea97bba835e` · 472c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I updated my cleaner's invoice template?

```text
Just got an invoice from our new cleaner, and it looks so unprofessional, it's full of spelling mistakes and formatting errors. I used to do this stuff for a living, so I am more than happy to send her a new one free of charge - honestly, I have to include her invoice in a letter to our previous tenant saying they're not getting any bond back, so I want it to look professional - but she hasn't asked me too.

Would I be wrong if I updated my cleaner's invoice template?
```

**Verdict:** KEEP

---

## 305 · `96b544b5ca09e956` · 787c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I refused to pay extra down-payment for an apartment?

```text
Recently applied for a 2 bed/1 bath apartment. It will be shared by myself, a female friend(I'll refer to as K), and her boyfriend(I'll refer to him as BF) At the end of the semester another friend of mine will move into my room and we will split costs 4 ways. Everything will be split as evenly as possible. Today we got denied the low down payment option of $400 because BF, who makes the most money, has shit credit. They wanted one months rent of $2199 as a down payment. I've got my portion of $2199 saved up but K and BF do not. My issue is I would not be willing to put down additional down payments past my 1/3rd of the quoted $400 because my credit is great and I've put effort into keeping it that way.

Would I be wrong if I refused to pay extra down-payment for an apartment?
```

**Verdict:** KEEP

---

## 306 · `9882a74a31886c66` · 656c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for not splitting gas/activites?

```text
Hi! So a long distance friend is in town, he's only ever usually around 1-2x a year. We typically try to meet up at least once and hang out. The past two times he's paid for whatever activity it is(I pay my own entry fee if applicable, and my own bill if it's food), and I haven't been helping to pay gas when we're out(he drives us all around usually, and last time drove me back to my place, a good hour away). So we're all talking about hanging out, but I have significantly less money than I thought I would. I feel kinda bad now that I'm learning to drive and discovering how much gas can be guzzled.

Would I be wrong for not splitting gas/activites?
```

**Verdict:** KEEP

---

## 307 · `9892eff1671e1f01` · 523c · ask=title

> **Title** (not part of the prompt): WIBTA if I tell the man on the bus next to me to leave?

```text
I am on a long bus trip. The man next to me is around 40, round aka fat build. I am doing some stuff for uni. minutes after the departure he pulls out a bottle of alcohol. Fine, alright. Then he fell asleep. Problem is that he is spreading his legs over to my space, he is snoring, he has bad breath. It is extremely distracting. Anyways, he is awake now. I don't think it is an appropriate way to behave oneself. Especially when he just took another sip.

Would I be wrong if I tell the man on the bus next to me to leave?
```

**Verdict:** KEEP

---

## 308 · `9921fd17fe389059` · 1192c · ask=body+title

> **Title** (not part of the prompt): WIBTA for asking my trans friend to stop copying my style?

```text
So I’ve got a really close friend who recently came out to me as MtF transgender. We’ve always gotten along really well, but since she’s started her transition Ive started to get the feeling that she’s been copying me - a lot. 

It started off with little things. She talked about really looking up to my style and asked to go thrifting with me. Of course I agreed. But over the past few months she’s started doing more and more of the same things as me- poetry, yoga, listening to the same music, watching the same shows. I wouldn’t mind if she was just taking inspiration from me or if she only did a few of these things, but I’m genuinely feeling like she is taking my entire personality. 

So far, I haven’t been able to bring myself to say anything to her. I know that transitioning is difficult and that imitation is the sincerest form of flattery— but I also don’t think it’s an excuse to copy every aspect of my style, interests, and personality. I don’t want to cut ties with her because she’s genuinely a really good person, but right now I’m honestly resenting her too much to think straight. Am I the asshole?

Would I be wrong for asking my trans friend to stop copying my style?
```

**Verdict:** KEEP

---

## 309 · `9937a4862dc9da33` · 544c · ask=body+title

> **Title** (not part of the prompt): WIBTA for requesting that my roommates to wash their hands after using the bathroom?

```text
They are both in their early twenties. My room is directly next to the bathroom, so I have no choice but to be aware of their hand-washing habits, and neither of them EVER washes their hands after using the bathroom, whether urinating or defecating, and it really grosses me out. I put up a fairly subtle sign about hand-washing about a week ago but that did not help, so now I feel like my only choice is to just confront them about it. Thoughts?

Would I be wrong for requesting that my roommates to wash their hands after using the bathroom?
```

**Verdict:** KEEP

---

## 310 · `998b8afbe4346187` · 1168c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told my family I don't want to move back home?

```text
I'm a senior in college and I will be graduating in a few months. My family lives an hour away from where I go to school and have been pressuring me to move back home after graduation. My relationships with some of them aren't bad but they aren't good either. The best relationship is with my mother who would give me her last dime if I needed it. My sister and father on the other hand are much different. They are essentially the same person in that they want to dictate everything, be in control, and they are never wrong. I'm definitely the black sheep of the family and when I do go home I feel like a visitor instead of a family member. Not because of how they treat me I just have different interests and passions than the three of them. I have mentioned this to my mother and it broke her heart so I haven't mentioned it again. However with graduation looming they keep asking when I'm moving back home but I'd honestly rather get a job and stay here. TL:DR I'm the black sheep in my family and don't want to move back home with some family members I don't have the best relationships with.

Would I be wrong if I told my family I don't want to move back home?
```

**Verdict:** KEEP

---

## 311 · `9a69372e41b2223d` · 1709c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I invited my dad to my wedding, but told him his girlfriend couldn't go?

```text
I'm not planning on getting married soon, but being a young woman I have a lot of thoughts about my future wedding, and I have a really bad relationship with my dad's girlfriend. She listened in to phone calls with my mom and reported it back to my dad, she has called me a "dumb bitch" for waking up early for school (which then woke up the dog and she insists that the dog sleeps with her so it woke her up too), makes digs at me for not doing good enough no matter what I do (for example, if I've been working a lot one week she will make a comment about how gross my room is, or if I do clean my room or do something around the house she will make a comment about how it wasnt good enough.). She hides food from me and gets angry if I eat anything that someone else may have wanted to eat and has called me a "fat bitch" for eating the last muffin once. She insists I buy my own food (which she then "borrows" and then calls me selfish for buying it for myself.). She has made several comments to my dad about how he "cant afford to take care of me" and how it "isnt his responsibility" (even when I was under 18, she would insist that I go live with my mom instead). I can't even stomach the thought of seeing her on such an important day In my life, she has made the last few years of mine miserable and seeing her would ruin it for me. I was thinking of just inviting my dad but specifying that she was not invited, and then letting him know that if he insisted she go then neither of them would be welcome at the ceremony, and then just having my mom walk me down the isle. Is this wrong? Is it an ass move?

Would I be wrong if I invited my dad to my wedding, but told him his girlfriend couldn't go?
```

**Verdict:** KEEP

---

## 312 · `9aa4a95faeb76e4b` · 567c · ask=body+title

> **Title** (not part of the prompt): WIBTA If I recorded the rude 6yo to prove to his parents that he is a rude child

```text
So I've been forced to watch the kid every so often for little to no pay since his somehow family ( I dont know how, possibly a family friend thing). I don't work but I am looking for work and brought up doing the babysitter thing but only for INFANTS, not a whole grown child. The kid is insane he curses and acts like he is the boss but when his dad is around he is the perfect gentleman.

Would it be wrong (morally and legally) if I recorded him and showed his parents?

Would I be wrong if I recorded the rude 6yo to prove to his parents that he is a rude child?
```

**Verdict:** KEEP

---

## 313 · `9b0a9e21213554b1` · 490c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for telling my roommate to bring his own TV next semester?

```text
I’m a college student with one roommate and bought a brand new TV for the dorm. I bought it knowing it’d be shared but still owned and paid for by me. We originally agreed I would bring the TV and he would bring his Xbox to share but my roommate had a tendency to hog the TV and use it late into the night while I slept. I plan on telling my roommate to bring his own TV next semester so I can use mine as I wish.

Would I be wrong for telling my roommate to bring his own TV next semester?
```

**Verdict:** KEEP

---

## 314 · `9bf247c5fdc2880b` · 1118c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I kept the $20?

```text
I am aware this is faintly ridiculous but here goes.... At the super market the other day I purchased a couple of items through the self serve check out. On payment I also got cash out of my account ($20) The problem is on gathering up my groceries I left the $20 behind. Went to work the next day where I realised my error, I phoned the supermarket (long shot I know) told them my story, and found out $20 was handed to the supervisor! Yay! Went after work today where I collected the money and chatted to the supervisor who told me it happens quite often, they keep it for a couple of weeks but most often the money isnt claimed. On arriving home I opened the envelope and saw with the cash is a duplicate receipt. On the receipt are items I didnt purchase and someone paid with cash, leaving their change. Now I'm really struggling with my conscience. I have gone from excitment that my money was handed in to feeling crap that I have someone else's money, and they might need that $20 really badly. I want to have not lost my money but does that justify taking someone else's??

Would I be wrong if I kept the $20?
```

**Verdict:** KEEP

---

## 315 · `9c19fcbafa76e8f9` · 930c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I limited someones toilet time?

```text
Ok simple question here we have a work experience guy coming through his school and to put it simply he's the worst person to work with. An ok guy but lazy and moody (the moods are fine it's the unfiltered attitude I hate).

So here's the issue: he goes into the employee bathroom (the only bathroom in store, we're in a mall) and takes and hour long "crap". Then does like 20 minutes of work and then another hour long "crap". Basically unless his organs are failing on a daily basis he's getting out of work. Now I know that some people take long in the bathroom heck I take like 15 minutes to do my business but this is insanity.

So would I be the ass if I limited to 20 minutes at most? Keep in mind other people need to use that toilet too and we can't leave him here because he's technically not hired by our company so sometimes people have to hold it for like an hour.

Would I be wrong if I limited someones toilet time?
```

**Verdict:** KEEP

---

## 316 · `9c4b005889d5612e` · 465c · ask=title

> **Title** (not part of the prompt): WIBTA if I kick my roommate out just because I got annoyed by him and do not trust him

```text
We are living together for 2 months. He is a normal dude, we do not have any complex communication problems. He knows that this apartment for him is temporary. 

But, he keeps asking me could he stay more. I’m generally okay with him but I can support this apartment on my own and I just don’t like him in general. He does not have a better option so I feel guilty.

Would I be wrong if I kick my roommate out just because I got annoyed by him and do not trust him?
```

**Verdict:** KEEP

---

## 317 · `9c87b42b6ec623ad` · 1092c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for asking my best friend what she plans to wear to my dad's inurnment?

```text
My dad will be interred at Arlington Cemetery in the next couple of months. My best friend is planning to attend. In the past she has embarrassed me before on three occasions wearing a black skirt and knee socks combo that in my opinion is too short for how she carries herself; you can see her bum if she bends over the wrong way etc. I hate to feel like a slut shamer or to be policing what women wear but it is personally not something I would ever wear as I feel it is too revealing (ironic as I work a side job that involves nudity/little to no clothing). Would I be a dick if I ask her what she's planning on wearing? Obviously it'll likely be black which is why I'm worried this too-short-skirt (again, imo) might make an appearance. It would make me feel anxious because I know for sure my mom and sister would be judging her for it (they are a lot more conservative than I am) and I'm worried I'd be focusing on that and getting anxious about it rather than being present at my dad's ceremony.

Would I be wrong for asking my best friend what she plans to wear to my dad's inurnment?
```

**Verdict:** KEEP

---

## 318 · `9d07bb21f69d3cda` · 1326c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I hung up on someone with an accent I couldn't understand?

```text
I've been shopping for cars. Took a test drive recently with a salesman with a very thick accent who was hard to understand. I didn't end up wanting the car and am still looking. I called a different dealership recently to ask about a different car and realized as I held for someone in the sales department that I was hoping that the person didn't have a thick accent. Then I wondered what I would do if they did -- would I be an asshole if they answered the phone and had a difficult to understand accent and I just hung up and tried again? 

This is on the heels of recently trying to report a warranty issue with a phone and the cell phone rep had such a thick accent that the entire call was me saying, "excuse me? could you please repeat that?" It was miserable and took twice as long. Then something ended up being wrong but instead of telling her, I wrapped up the call and called back (I got someone else with an accent but I could understand him so it was no problem). 

Context: I am a mid-thirties female who is fairly well traveled (visited 27 countries) and love other cultures. But if I'm completing a transaction with complicated information, I really want to be able to understand the person I'm working with.

So, asshole or no?

Would I be wrong if I hung up on someone with an accent I couldn't understand?
```

**Verdict:** KEEP

---

## 319 · `9d621e8cd7977225` · 1574c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I called out my grandma for giving the family pets too much food?

```text
To start off, one of the cats in the house are fat. Really fat. Every time I walk into the kitchen, she runs over to her food dish and starts meowing like crazy, as if she hasn't been fed in a week, yet she gets fed breakfast and dinner. And the worst part about her is that her shit STINKS BAD. HORRIBLY BAD. I'm talking the FBI could synthesize the smell, and use it to clear rooms.

Yet, my grandma seems oblivious to how obese she is. When she feeds the fat cat in the morning, she'll give the cat a Mount Everest sized pile worth of food, so when she's not looking I have to split it up between the two food plates (we have two cats). Before we leave for school, she'll give the cats treats, and then the dogs treats. I asked why do they get treats so much, and she'll say "I'm feeding them??" First of all, treats are called "TREATS" for a reason. They are to REWARD your pets for good behavior, training them, etc. She just gives treats to our dogs just because, not for good behavior. She also likes to feed them air-popped popcorn a lot, I mean a lot. I'll be sitting watching TV with her while she's eating popcorn and toss a piece of popcorn to each dog at least every 5 to 15 seconds (I counted with my fingers, 1 1000, 2 1000, 3 1000, etc). Based on a search, air-popped popcorn isn't harmful, but she puts butter on her popcorn, which is fattening.

I'm starting to get afraid that before we know it our animals are going to have obesity problems. How should I handle this?

Would I be wrong if I called out my grandma for giving the family pets too much food?
```

**Verdict:** KEEP

---

## 320 · `9dadcfefac370945` · 1370c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for going back on an agreement to let my friend keep living with me?

```text
I have a little two bedroom house; for all the time I've owned it I've rented out my spare room to lodgers. Back in May, my friend asked me if he could move in for the summer because he wanted to move to our town and get a job before moving in with a mutual friend in September. I've tried to avoid living with friends but, as it was just until September, I said it was fine. Since he moved in in May, he's been an awful lodger. I'm not going to go into it all here but he is always late with rent, hasn't helped out barely at all since he got here 5 months ago (ie; cleaned the bathroom once, that's it). He's not managed to find anywhere to live with our friend and she's been put off by my experience so doesn't want to live with him anyway. A few days ago, after a while telling me he'd be out by November 1st, he said his backup accommodation had fallen through so could he stay here for a few months more, or until he finds somewhere else to live? I panicked and said yes but, on reflection, I don't think I can cope with any more of this uncertainty - or even living with him, to be honest! On the one hand, 6-7 weeks is plenty of notice and he has been an objectively awful lodger, but on the other hand he is my friend and I don't know what he'd do if he couldn't live here.

Would I be wrong for going back on an agreement to let my friend keep living with me?
```

**Verdict:** KEEP

---

## 321 · `9edef8e6fbcc30ee` · 1261c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told my family I feel unwanted?

```text
Title says it all. Recently I’ve been feeling as if I’ve been ignored by my family. When I go to tell them about a story idea I have or talk about something I like in general they huff and puff or groan or just straight ignore me. It makes me feel as if what I want to say is unimportant if it isn’t feeding into a constant stream of complaining about my dad. My mother will spend hours at a time watching videos on some K-Pop band, and listening to my youngest sister practically beg for tickets for one of their upcoming shows even though we can’t afford it. When it comes to my other sister it’s more understandable since she’s going off to college soon. But I just feel like none of them want to hear what I have to say, at all. I’m currently in college (yay being a broke ass college student /s) and I’m set to graduate and move to a better college this semester, but it feels like that’s just... irrelevant now. Last time I brought up how I feel over this, I made my mother cry and I was told I was being a selfish cunt. I don’t want to hurt any of them, but I also want to connect with them and share my passions with them. Or rather, makes me feel like my interests are unwanted/pointless and dumb?

Would I be wrong if I told my family I feel unwanted?
```

**Verdict:** KEEP

---

## 322 · `9ee1606275353c0c` · 883c · ask=title

> **Title** (not part of the prompt): WIBTA if I told on a coworker

```text
So I (18M) work at a call center for a medical clinic. Today, I was sitting with two of my coworkers discussing work and the topic shifted to some other business ventures the company has made. The woman said “these Jews just want to make money”, in reference to the two owners, who are both Jewish and pretty well renowned in their field of medicine. While I am Jewish, I’m not very threatened or offended by this, however I feel somewhat compelled to not let this slide.

My questions is, would it be the right thing to speak to a manager about this, exposing myself as the one who told and potentially getting her in trouble, or letting it slide as an offhand comment? For context as well, I have been there 8 months and she is relatively new, so 3-4 months. She has also made many nagging comments about coworkers, patients and managers.

Would I be wrong if I told on a coworker?
```

**Verdict:** KEEP

---

## 323 · `9f26f62e4fea4695` · 1017c · ask=title · scrubbed

> **Title** (not part of the prompt): Wibta if I take my neighbor’s free gift from a cable company? (Light hearted)

```text
The house next to mine is almost 100% abandoned, they left two and a half months ago leaving their garbage can overflowing on the street. That was in winter and the heavy winter winds blew the can over and the garbage collectors passed it by, unable to use the trucks arm to pick up the tipped can. After a few weeks I went over before the truck came on garbage day, righted their can, and picked up their trash. The collectors then took the garbage. The reason I don’t think their house is all the way abandoned is that someone came by and put the can back on the side of the house. Since then there has been no activity at their house. It was an older couple and I’m guessing they went to live with their children, maybe. Today I went to load some stuff into my car and noticed a cable company left a flier in a bag with a metal straw. I’m moving out of my house today and will only be back to clean and sell some furniture.

Would I be wrong if I take my neighbor’s free gift from a cable company? (Light hearted)?
```

**Verdict:** KEEP

---

## 324 · `9f90c741ce72ac0e` · 629c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for hanging out with a person one of my close friends hate?

```text
I will be regarding the person inviting me to the party as L  The person that hates L with be S L has invited me to a party with some of her other friends. L and S were both best friends but now S hates L because of a "toxic relationship". I'm not really sure what went down but I want to stay out of it. I love S as a friend and I'm also having a fun time hanging out with L. If S finds out that I've been hanging out with L she will most likely flip out at me and stop hanging out with me, or just put me at a distance, which I don't want to happen.

Would I be wrong for hanging out with a person one of my close friends hate?
```

**Verdict:** KEEP

---

## 325 · `9fdd24041a0527da` · 396c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I report my upstairs neighbor to the complex?

```text
I just moved into a new appartment and my upstairs neighbors, who I have never met, are VERY loud and smoke pot pretty much every day. The smell comes through my vents and if I have the screen door open, it is very obvious. They regularly have screaming matches filled with profanities and banging and stomping around on my head.

Would I be wrong if I report my upstairs neighbor to the complex?
```

**Verdict:** KEEP

---

## 326 · `a04f5dc65374d797` · 1397c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for reducing contact with friends who excluded some of our group from a trip?

```text
At my school, it's a tradition for seniors to take a beach trip right before graduation. My group of friends all seemed pretty excited about it, and we had been talking about it for a while, but nobody had made concrete plans. However, a number of us have been explicit in our intent to go on a trip. In my group of 8 friends, there are 3 people, "Clara, Jane, and Emily", who will often exclude people from things that could easily be group activities. For example, if a group of 5 of us always watch a TV show together, Jane will only text 4 of us, excluding the same girl every time and forcing us to text her individually. This sort of thing happens a lot, especially when Clara and Jane are involved. Four days ago, I texted the whole group to get started on making plans for the beach trip. Suspiciously, everyone replied to my message except Clara, Jane, and Emily. Finally, Clara texted me today to say that a few weeks ago, her dad had paid for a hotel room that would only fit 3, and those 3 people would, of course, be Clara, Jane, and Emily. I understand that obviously, our group of friends would be too big to all fit in that room. But I don't see why the three girls couldn't have *at least* included us in the planning process instead of waiting weeks to let us know what was going on.

Would I be wrong for reducing contact with friends who excluded some of our group from a trip?
```

**Verdict:** KEEP

---

## 327 · `a0cff719854f4569` · 588c · ask=closer

> **Title** (not part of the prompt): UPDATE: WIBTA if I humiliated this choir because of posts they made about me on social media?

```text
Yesterday, I made a post asking if I would be the a-hole if I was to screw up this choir's performance. Today was the first rehearsal for the piece we're performing. When I got there, I talked with them about the posts they made about me. I gave them a description of how the posts were hurtful. I also told them that I would NOT substitute their fifth voice. 

After a thirty minute discussion, they apologized to me and agreed to take down the rude posts. They were also ok with the fact that I wouldn't participate.

Thank you to everyone who responded to the post.

What would you do?
```

**Verdict:** KEEP

---

## 328 · `a11f9983d2b9dbb2` · 400c · ask=title

> **Title** (not part of the prompt): WIBTA for charging high amounts for snacks.

```text
So at my school teachers don’t care if kids sell stuff between them. Would I be a asshole for charging 3 dollars for a item that costs me 1 dollar. Last year I had it down to science raking in about 40 bucks daily profit but I feel like a ass overcharging them. Most of them are from very wealthy families who give them a lot of pocket change.

Would I be wrong for charging high amounts for snacks.?
```

**Verdict:** KEEP

---

## 329 · `a22969dca08add1e` · 1000c · ask=title

> **Title** (not part of the prompt): WIBTA if I installed a plastic cage over the thermostat?

```text
I live in a house with two other people.  The agreement was in exchange for exclusive use of the garage and driveway I would pay the electric.  

It worked out fine in summer and fall, but since it’s gotten cold the electric bill has gotten ridiculous.  Last month it was $162.  I’ve talked to the roommates about it and trying to cut the cost down.  They agreed, but roommate’s gf who is here almost every day is not following through.  She cranks the heat up and leaves it that way.  Even when she leaves. 

I talked to roommate about her doing this and he said he would talk to her again, yet today when I came home the house was empty and the heat was on 85.  I don’t want to cause drama with her so I’m thinking of installing a plastic cage over the thermostat that I would have the key to.  I wouldn’t let the house get freezing, but it would also prevent someone who doesn’t even live here from driving up the electric bill.

Would I be wrong if I installed a plastic cage over the thermostat?
```

**Verdict:** KEEP

---

## 330 · `a2495b27510232cd` · 960c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if i avoided talking to a good friend who recently expressed some pretty fucked up things about trans people?

```text
So I attend an all boys school, and just today we had a talk given by a transgender author and journalist. It was great, and the speaker was really good, but afterwards quite a lot of people were being pretty cruel about her and trans people in general (as usually is the case). The guy in particular has been a really close friend ever since I came to the school, and he's really smart. So it was a huge surprise when after the talk he told me that she (the speaker) was a 'cocky bitch' who should be 'fucking put down'. Ever since he said that I've avoided talking to him as I think tht it's pretty disgusting that he thinks that way, and i don't know if I can still be friends with him knowing his views on the topic. However I feel like I'd be overreacting by avoiding talking to him, considering how good of a friend he can be.

Would I be wrong if i avoided talking to a good friend who recently expressed some pretty fucked up things about trans people?
```

**Verdict:** KEEP

---

## 331 · `a2ad938a27655715` · 466c · ask=body+title

> **Title** (not part of the prompt): WIBTA for giving small tips to servers (when I make less $ thant hem)

```text
I live in a state where servers have to be paid minimum wage. I earn minimum wage in a customer service position and can't accept tips.

Should I be expected to tip more than 10%? The servers are likely earning 2x as much per hour as I am. I have to perform similar duties since I'm a courtesy clerk (taking back bad food, chatting up customers, providing a warm & fuzzy feeling etc)

Would I be wrong for giving small tips to servers (when I make less $ thant hem)?
```

**Verdict:** KEEP

---

## 332 · `a3f85e2ae3612b46` · 918c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I quit my job

```text
I (22M) have been at my current job for about 9 months. Part of my job is spent in an office, and part of it is spent in our emergency department. I love working in the emergency department but over the course of working here my job satisfaction has gone way down, to the point of dreading going to work & praying I'd get T-boned on the way there. Yesterday, my boss spoke to me in a manner that was pretty derogatory but not entirely undeserved, and I informed her that I'd no longer be accepting shifts in the office as of next week, though indicated I'd be happy to continue work in the emergency department. My boss wants to have a meeting between her, my boss in the emergency department and myself, and I've indicated I'm happy to attend. I'm currently compiling a list of things that would make my work conditions more tolerable but would ideally only want to work in the ED.

Would I be wrong if I quit my job?
```

**Verdict:** KEEP

---

## 333 · `a409402456296dfd` · 1993c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I confront my sister about her friend

```text
So last night my sister invited me out drinking with her and some friends of hers (acquaintances of mine) and we had an overall good time and it was a blast, however something from last night is just bugging me. My sister's friend has, in previous encounters, expressed her interest in me, and it was mutual. However, my sister told me that her friend has a lot of baggage, and that she'd prefer if I didn't pursue her. No problemo, I'll respect her wishes and just drop it. From previous encounters I know this girl has a lot of issues with trusting men because she's had a bad stretch of guys and I feel for her on that. Here's where the issue of last night comes in, one of my sister's friends is a shameless womanizer. He's a great guy and I enjoy his company, but he is very honest in that he likes to date and sleep around (no judgement there I've done the same). My sister has also previously had a fling with the guy. So my sister was progressively pushing them together over the course of the night and by the time I left to go home they were making out in the bar. I'm concerned that my sister has set her friend up to be hurt by a guy that she knows is a womanizer when she knows her friend is looking for an actual relationship. While it technically is none of my business it's the kind of behavior she's exhibited in the past where she's unintentionally set a friend up to get hurt by trying to play matchmaker and not taking into consideration the consequences of her actions. I would like to confront my sister about this kind of behavior and how it can it be toxic not only to her friend but also to her friendships with people and her friend circle. I'm torn on this one TLDR: Sister set up her friend that wants a relationship with a guy who is not really a relationship kind of guy. I think she didn't think this through and just set her friend up to get hurt. I want to tell her that but am unsure of if I should.

Would I be wrong if I confront my sister about her friend?
```

**Verdict:** KEEP

---

## 334 · `a433a5a637f7c850` · 1403c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA For getting mad at a friend that applies for the same jobs as me?

```text
This post is more for my roommate since it was the job he applied for but here's the backstory. I have two roommates we will refer to them as Chip and Dale. Chip, Dale and I are all about to graduate college this semester and are applying for jobs. Chip and Dale are the same major and have many classes together. For the last year or so whenever Dale would apply to internships and jobs, Chip would go and apply for the same jobs, sometimes even without telling Dale. It was not too much of an issue before because neither would usually get these jobs/internships. Recently Dale told us about this job that he really wants and how it's one of the only jobs he's applied for that he can really see himself doing. Chip then went and applied that same day, which was also the deadline for the application. A few weeks pass and Dale finds out that he didn't make it to the next round of interviews. Chip on the other hand found out that he's getting flown into their headquarters for an interview and chose to gloat in front of Dale. Dale does know what to think of Chip applying to jobs that he is applying for. I think that since Chip and Dale are the same major from the same school, just with different GPAs, Chip's application directly affected Dale's. Also he should not be competing with his friends for employment.

Would I be wrong for getting mad at a friend that applies for the same jobs as me?
```

**Verdict:** KEEP

---

## 335 · `a522d60741dc2711` · 703c · ask=title

> **Title** (not part of the prompt): WIBTA if I ask her not to bring her kids on our dates?

```text
I’ve been seeing a woman for around three months now give or take.  We both have children from previous relationships.  Things have been going really well.  

The last two dates however she brought her young children with her.   One was at a restaurant and one was at my house.  I understand that she wants to see if I mesh well with her children, although it does seem really soon to me to meet them.  But we are really still in the getting know each other phase ourselves and when she brings her children they are the focus of attention (as they should be).  It’s not really giving us a chance to connect and interact with each other.

Would I be wrong if I ask her not to bring her kids on our dates?
```

**Verdict:** KEEP

---

## 336 · `a54f7799cd274d40` · 778c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I didn't include my sisters in my bridal party?

```text
I have a good relationship with my sisters, but it's not like that close, sibling-best friends kind of relationship that some people seem to have.  I have 2 best friends that I absolutely want in my bridal party, and then 2 sisters and 2 other close friends that I am more indifferent towards.  I don't want to be one of those people that has a crazy amount of bridesmaids, but I honestly don't know how to prioritize who should and shouldn't be in the bridal party. I feel like all my friends have always had their siblings in their bridal parties, and I'm just not sure if I would be risking offending them if I didn't include them.  Is it considered rude to have a bridal party that doesn't include family?

Would I be wrong if I didn't include my sisters in my bridal party?
```

**Verdict:** KEEP

---

## 337 · `a56a5d368f743b7f` · 379c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I asked if I could buy my neighbors car?

```text
My neighbor is a really nice lady and had a red Mercedes that her son wrecked. He drifted into a lamp pole and it ruined a tire and the back right side of the car. 

I asked my parents if I could ask if I could buy it from her and they said sure but it would be rude of me. 

Would it be rude? Am I missing something?

Would I be wrong if I asked if I could buy my neighbors car?
```

**Verdict:** KEEP

---

## 338 · `a5d3566e24b817dd` · 1704c · ask=title

> **Title** (not part of the prompt): WIBTA if I get my neighbour ticketed for parking in front of my house?

```text
To preface, I don't use street parking at all.

I'll cut to the chase. I don't like it mainly because it's a beaten down lifted (really high) truck. Had he parked a SUV, normal truck, mini van, or sedan, I wouldn't have cared at all. One of my other neighbour park their Hyundai SUV there too, and I don't care at all. 

A couple of things really ticks me off. His drive way is empty during the day but he doesn't want to move his truck onto his own drive way. At night, he parks this truck horizontally across the sideway in-front of his house because there is no street parking between 2AM and 7 (or 8) AM. (he has two other cars parked on his drive way) I hear him moving the truck around 12-1AM every night. It's loud. (Not a concern of mine) The height of the lifted truck also obstruct the view of the front of the house as well as the street which can cause slight difficulty reversing my car in the morning. 

There are two violations from the city bylaws that I've found.

1. in front of or within 1.5 metres of the entrance to a driveway or so as to
prevent ingress to or egress from such driveway;

2. for a longer period of time than 3 consecutive hours;

The reason I'm posting is because both violations doesn't effect me. If they did, I would'nt've hesitated about calling the bylaw officers. His truck being inches away from my drive way exit doesn't really affect me too much. I can deal with this. I did a rough measuring and it's about 12 inches away from my drive way. I'm capable of reversing out of my drive way without any trouble. Also I don't care if he park how many hours. It's just THAT truck.

Would I be wrong if I get my neighbour ticketed for parking in front of my house?
```

**Verdict:** KEEP

---

## 339 · `a9249b40778af2b5` · 655c · ask=title

> **Title** (not part of the prompt): WIBTA If I tell my brother to stop bringing so many people over for family dinner?

```text
I have 3 brothers and my family are really close we eat dinner every sunday together, it's a tradition that we all just take part in even though we all moved out. My brother travels with a literal entourage like something out of a show and for the past 6 months or so he's brought about 4 or 5 girls and sometimes a couple guys but mostly all girls to these family dinners. My other brother brings his 1 girlfriend, I bring my 1 girlfriend I don't see why my other brother needs to bring so many people with him everywhere. I feel like saying something but idk

Would I be wrong if I tell my brother to stop bringing so many people over for family dinner?
```

**Verdict:** KEEP

---

## 340 · `a94382639cb45e66` · 665c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if everytime I get a netflix new sign in email I ask one of my friends if it was him?

```text
I gave one of my friends my netflix account password. The thing is he’s not the only one using it besides me, but I have a hard time believing my dad(he’s the other person who uses it) is constantly signing in on new devices. I just have this strange gut feeling that my friend has given out my password to someone else because of the many new sign in mails I’ve been getting. But I just feel like an asshole for only assuming it’s him and constantly asking if it was him who signed in. This whole dilemma makes me feel like a control freak and a little guilty.

Would I be wrong if everytime I get a netflix new sign in email I ask one of my friends if it was him?
```

**Verdict:** KEEP

---

## 341 · `a961c1a388dfddd7` · 1007c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I avoided mouth contact with someone because she had herpes?

```text
So I've been hitting it off with this girl for awhile. We've known each other through the interwebs for a few years. She's going to move to my state, and we decided tk start hanging our irl. She sent me an unsolicited underwear pic at one point to things are heating up. Only problem is, she has mouth herpes, and I'm kinda torn on what the ettiquite is for a situation like this. On one hand, I want to play it safe and avoid an incurable STD. On the other hand, to put it nicely, she's not the most stable mentally, and pretty much had a breakdown when she initially learned she had herpes, which I had to support her through, but I really do like this girl and don't want to inadvertently hurt her feelings by avoiding mouth contact. I haven't brought this up with her yet. This dilemma is entirely in my head at this point. Or am I just the asshole for making these assumptions and overthinking things in the first place?

Would I be wrong if I avoided mouth contact with someone because she had herpes?
```

**Verdict:** KEEP

---

## 342 · `a9a97c6e87bbfff8` · 1658c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I stopped talking to my long time friend?

```text
So this friend and I have been friends for roughly 10 years now. We were never super close but they've always been super open with me and always come to me when they need an unbiased person to talk to. I had fallen out of touch with them for a while but we reconnected last year when we found out our friend was dying. After the funeral, they had a really hard time coping, especially since they had been really close to our late friend. A couple of friends and I tried to be there for them, but it was really hard because we were also grieving. Although I know people grieve in different ways and for different lengths of time, I don't know what else to do. That was almost a year ago, but it doesn't seem that they're doing much better. Whenever I spend time with them, they start complaining about their lives and they eventually bring up our old friend and how hard our was to lose them. I didn't mind it as much, since I've always been more of a listener, but after a while I realized I can never talk to them about what I'm going through either. When our friend passed away, I was the one comforting them but never once did they do the same for me. I brought this point up recently, but they told me that I didn't need much comforting anyways since I wasn't as close to them as they were. It was really upsetting to hear them say this because, although I may have not known our friend for as long them, I had still been their friend too. I'm not sure if they said this because they're still grieving, or if they meant it,  but after this I realized I was done with being friends with them.

Would I be wrong if I stopped talking to my long time friend?
```

**Verdict:** KEEP

---

## 343 · `a9cdc78f58c2c379` · 427c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I just completely leave the family supper?

```text
Ok, so this is kinda simple, but my extended family and I are on vacation on a ranch, which has a few buildings. Supper has been taking longer and longer to be made, with a steady decline in the quality of the food. Currently it has been taking until about 8:45, which is when I start to get exhausted, and I have been thinking of just leaving and going to sleep.

Would I be wrong if I just completely leave the family supper?
```

**Verdict:** KEEP

---

## 344 · `aa084fbf251f3234` · 690c · ask=title

> **Title** (not part of the prompt): WIBTA if I called animal control to pick up my neighbor's cat?

```text
So my neighbor's have two cats. They say one is an indoor cat and the other one is an outdoor cat. The outdoor cat is very sweet and I don't mind her hanging out on my porch for some shade. She's started bringing dead animals on the porch and just leaving them. I talked to my neighbors and they just said "we can't control what she does outside."
At this point I'm annoyed at them and not the cat. I've never once seen them feed her or even let her inside when it's pouring rain. I want to call a shelter to come get her since she's becoming a problem for the neighborhood, and because I'm worried she'll get hurt.

Would I be wrong if I called animal control to pick up my neighbor's cat?
```

**Verdict:** KEEP

---

## 345 · `aa74820931906261` · 1786c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for refusing my grandmother's "inheritance"

```text
My grandmother recently told me she wants to leave me her house when she passes. I'm not a fan of this for a number of reasons, but I said it was fine. She's stubborn and typically gets what she wants, so I figured it was best to accept and figure it out when the time comes. However, this eventually turned into "I want to transfer the property to your name now so that the nursing home can't take it when I go in", and now I really don't want to have anything to do with it. I've tried explaining to her that the nursing home can't seize her house, and that Medicaid would only place a lien against her house if she can't pay for her care and it's within 5 years (which is not likely to happen). She's in good health and it will likely be several years before she has to go into a home. During that time, the house will be in my name, but she will live there and continue to pay the mortgage until such a time that she needs to go into a nursing home. I don't see the point to it being in my name. If anything happens to the house, I'm liable for it. If the worst case scenario happened and she needed the cash sooner sooner than later, I'd sell the house and give her the money anyways. Even if neither of those things happen, at best I'm left with a $80k mortgage on a house that I don't want and can't afford, so I'd still sell it. She's still under the impression that I'd move in once she goes into a home, despite my insistence that I would likely no longer live in the same state and have no desire to own property in her city. She still wants to do this her way, and I'm incredibly uncomfortable with it. I'd I do, there's a good chance she won't leave me the house in her will, which at this point I'm okay with.

Would I be wrong for refusing my grandmother's "inheritance"?
```

**Verdict:** KEEP

---

## 346 · `aacd4a8347db0427` · 1382c · ask=title

> **Title** (not part of the prompt): WIBTA if I abruptly quit a job at a family business

```text
So for some context, my uncle owns about 50ish prepaid phone stores. 
(This isn't really relevant, just giving a full picture)
I needed a job, and I needed one fast. He offered me a job and I took it, no questions asked 
(Regretting that now) for starters, the job is a bit of an incontinent commute. About 35-45 minutes depending on traffic. Which may not sound like a lot, but on minimum wage that's murder on gas money. The working conditions are pretty decent, I work in the back mostly putting in orders, keeping track of shipments, handling email's, etc, etc. Here's the problem though. Since I'm technically not officially on payroll, so I don't get direct deposit. My paycheck has been late on more than one occasion, and as of today it's two weeks late because the stores accounts are negative. I'm on vacation right now, so I'm out of state. That being said, I'm owed two weeks worth of pay. I'm a broke 20 year old and can't put up with inconsistencies like that. So when I get back home, I'm thinking about just quitting right then and there and finding another job. It would be a lot easier if it weren't family, but here we are.

TL;DR
I took a job my uncle offered me out of desperation and have had my paycheck be late one too many times, but I'd feel like an asshole quitting a family job so abruptly

Would I be wrong if I abruptly quit a job at a family business?
```

**Verdict:** KEEP

---

## 347 · `aae31fa1d263ec29` · 1031c · ask=body+title

> **Title** (not part of the prompt): WIBTA for asking my roommate to eat more quietly?

```text
I’m currently at uni, and I’ve managed to get a pretty good roommate. He’s really nice and chill, but of course we don’t live in perfect harmony. One thing that can kind of bother me sometimes is how he eats. It feels like he drags out his bites, and he makes it so you can hear *everything*, even the long, comical swallow. He also makes this soft moaning sound and breathes heavily when he eats. It seriously sounds like he’s pleasuring himself with food sometimes. It can be really distracting.

I feel like I shouldn’t ask him about it though, as it could be embarrassing. I know I have habits or ticks that I either don’t notice or I would be embarrassed to be told about. I also feel like he’s a lot better of a roommate anyway. I’m messier for sure, and a lot of times I can be up late on my computer (brightness down of course), which he assures is fine. I could always just go to the library too, but sometimes I just wanna study in my room. What do you think?

Would I be wrong for asking my roommate to eat more quietly?
```

**Verdict:** KEEP

---

## 348 · `ab8dc6f2ebdbb5b7` · 952c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked my boss for comp time to make up for being asked to work 7 upcoming Saturdays?

```text
Our company has decided to be a sponsor for a bunch of 5k races coming up and most of the planning has fallen on my shoulders. I don’t mind, it’s fun work. Issue is, we need staff members to run our sponsor table at each race and so far we don’t have enough volunteers. I don’t blame them, they’d have to be set up by 7am. That means waking up at 6, or earlier, on a Saturday or Sunday morning and working until around noon. We’re all paid a salary, so legally our boss can require us to work overtime with no extra compensation, but I think that we should get comped a 1/2 for every race because it’s a lot to ask people. And I’m honestly thinking about myself a lot here because I might get stuck doing all 7 events, and one is on a holiday weekend, meaning that I won’t be able to go down to the beach with my family like I usually do. So...

Would I be wrong if I asked my boss for comp time to make up for being asked to work 7 upcoming Saturdays?
```

**Verdict:** KEEP

---

## 349 · `ac705e5cd7df80f7` · 2179c · ask=title

> **Title** (not part of the prompt): WIBTA if I tell a buddy that no one wants his girlfriend to come to game night?

```text
My group of friends have a once a month game night.  It’s mostly just the five of us getting drunk, playing cards or board games, and catching up.   It’s probably sad, but it used to be the highlight of my month.  

One of my friends has a girlfriend who for the last four times has tagged along.  None of us minded when he first asked if she could come, but she’s made these get together completely un enjoyable.  

She’s nice enough I guess, but she’s insanely annoying.   She always has to be the center of attention.  She talks about herself and her opinion almost nonstop and she’s really not that interesting so it doesn’t make for good conversation.  If someone is talking, she will interrupt and find a way to bring it back to her.  If the person keeps talking she will just talk over them.  I’ve sat there and watched her do it and raise her voice on purpose just to be louder than the person talking.  

Another thing that’s been awkward for me personally is that she tries to make me a stereotype.  I’m gay.  I’m not the “gay bff” type of guy though.  Yet she keeps trying to make me into hers.  She’s always getting weirdly touchy feely with me.  She tries to have “girl talk” with me.  One time she learned that I was going out to a gay bar and shows up uninvited and basically followed me around and ruined my entire night.   I’ve tried to be tactful with her since she’s my buddy’s girl, but she does not get the hint.  One time I went so far as to tell her that I don’t have any female friends because I just really only get along with guys (lie), but in one ear and out the other.  

No one except her boyfriend wants her there.   The rest of us have all talked about it.  None of us like her.   We just don’t know how to tell him.  I’m a rip the bandaid off kind of guy and just want to take him out for a beer and tell him.  I would be nice about it.  I already know the gist of what I would say.  “The group dynamic has just changed with a girl in the mix.  We’d like to make it a guys only night and maybe get together with you guys as a couple a different night. “.

Would I be wrong if I tell a buddy that no one wants his girlfriend to come to game night?
```

**Verdict:** KEEP

---

## 350 · `ac767fdf31978b2e` · 1150c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I canceled my friend’s plane ticket?

```text
I bought my friend a plane ticket on miles a while back because we were taking a trip together. A family discovery came up that made him want to push his trip until after I go on the 30th of this month - we discussed it and it seemed like the best plan. I felt a little used by this, but he was going to be spending time there after I left, and he didn't ask for this to happen. Annoying, but fine. (That full story is in another post on my profile if you really care to read it). Then his grandfather died. I got radio silence for a day and then he came back and has been pretty sparse with communication. I keep calling him because if he doesn’t want to come on this trip we need to cancel the flight or move it asap. If we’re going to move it, I need dates to move his flight to because I'm not trying to pay the fees twice, and he can't afford to. He gave me a range of dates that he'd been looking at, but I asked if he’d be willing to come to a big event with me a few days after (and push the flight a few days) and then, radio silence. I fly out tomorrow and need to move this flight.

Would I be wrong if I canceled my friend’s plane ticket?
```

**Verdict:** KEEP

---

## 351 · `acb92a27a01e5dc2` · 1515c · ask=title

> **Title** (not part of the prompt): WIBTA for calling animal control on my neighbor with 7+ dogs?

```text
Necessary background info, please read all of it before passing judgement

- I live in an area of unincorporated Los Angeles County (California), which means the legal maximum number of dogs is 4 per household/property unless its a business/rescue operation kinda deal

- The neighbor lives on around a half acre of land, the dogs have plenty of room to run around it, no obvious health hazard, all seem well fed but not fat. Its 3 large dogs (Shepheard mixes) and 4 small dogs (2 chihuahua mixes, 2 or 3 fluffy thing mixes). 

- Its not even the number of dogs that gets me. The problem is they have inadequate fencing made of vertical steel slats, so that the bigger dogs are well contained, but the small little yappy fluffy shit heads can walk right through onto the street and harass people walking by, wander into peoples front yards and bark at them, or in an extreme case, slip through a gap in the neighbors across the street's fence, and get in a fight with their 2 german shepherds. I managed to break it up and the little yapper ran back through both fences home, but i haven't seen it since, which was 2 weeks ago, and is why I say they have 2 or 3 fluffy dogs.

My main worry is if I called animal control to report the issue about the improper fencing allowing the dumb little shits out, animal control would get angry about the 7+ dogs and make the neighbor get rid of some, which is the thing that I might be an asshole for.

Would I be wrong for calling animal control on my neighbor with 7+ dogs?
```

**Verdict:** KEEP

---

## 352 · `adcb6aecaba719bb` · 790c · ask=title

> **Title** (not part of the prompt): WIBTA if I got mad at my friend for constantly flaking out on plans?

```text
So I have a good friend that I enjoy hanging out with, but they aren’t good at turning people down so whenever I invited her somewhere and she doesn’t want to go she says yes and then either constantly reschedules, claims she will get back to me with a time that works for her, or cancels at the last minute with some lame excuse. So would I be an asshole if I confronted her and told her to cut the crap and just be straight with me so that I don’t waste the time and energy trying to make things work if she doesn’t want to go? Also do I have any right to be upset since even though my gut tells me she is just being flaky, she is also a very busy person so there is a chance that I’m just overly sensitive.

Would I be wrong if I got mad at my friend for constantly flaking out on plans?
```

**Verdict:** KEEP

---

## 353 · `ae7161d0d7020c51` · 844c · ask=title

> **Title** (not part of the prompt): WIBTA if I set a limit to how much my mother in law can visit?

```text
This is complicated.  We live in his mother’s old house.  She gave it to us when she moved for work.  I was blown away by the gift, but i didn’t know what I was getting into.  

She visits almost every weekend.  I’m not exaggerating.  She just shows up and stays until Sunday.  I have two kids and she’s an awesome grandmother to them so I understand her wanting to be around them.   But holy fuck is she getting on my nerves.  She always has some unasked for advice or some complaint about the way we are keeping the house.  It’s driving me crazy.  

I want to ask the husband to stop letting her come so often.  Maybe once a month, but no more every weekend.   She gave us a house though and I know it’ll be a tough sell.  He’s a mama’s boy and likes having her here.

Would I be wrong if I set a limit to how much my mother in law can visit?
```

**Verdict:** KEEP

---

## 354 · `ae7ac603a329e49f` · 512c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I confronted our roommate about having her male guest stay over? I mean I just came home at 3:00am to see a strange man sleeping in my living room.

```text
I don’t know this guy. I live here too. Whose to say he’s not the reason my laptop is missing?! Why do I keep coming home to strange men in my house?! Is informing and asking other common household members of your guest not common curtesy? Is a quick text so hard? Let me hid my stuff like give me a notice!
Anyways would I be wrong for that?

Would I be wrong if I confronted our roommate about having her male guest stay over? I mean I just came home at 3:00am to see a strange man sleeping in my living room.?
```

**Verdict:** KEEP

---

## 355 · `ae9ded1d60aff814` · 1033c · ask=body+title

> **Title** (not part of the prompt): WIBTA for conpletely cutting my annoying friend off when everyone wants the group to stay together?

```text
Last Friday, my former friend (we’ll call him Ebert) shared that our one female friend made out with a guy to a guy said female friend hates. We told him to get lost, and I assumed this was permanent. I blew up on Ebert (i hadnt really been fond of him for a while, he’s a huge narcissist who changed his whole personality to fit in with the cool kids) and called him a narcissistic loser and things of the such. Fast forward to today, and it turns out everyone just got over it. I didn’t really care about the prior incident, it just pushed me over the edge. I tried my best to ignore him, but he kept trying to irritate me. He told everyone about me blowing up on him. All of my friends are telling me to just get over it and make up. I do this, as he’s moving in a few months anyway and I want to take one for the team, but I still hate him. What should I do

(Sorry if this makes no sense I’m shit at telling stories)

Would I be wrong for conpletely cutting my annoying friend off when everyone wants the group to stay together?
```

**Verdict:** KEEP

---

## 356 · `af5bfba079606e33` · 430c · ask=title

> **Title** (not part of the prompt): WIBTA for throwing a snowball at my neighbor?

```text
I was fooling around outside with one of my neighbors when he started throwing snowballs at me. It quickly got annoying, so I politely asked him to stop. He then whipped a big one at me, which missed. I chucked one at him but he ducked, and the snowball hit him in the face. He started crying and ran to his house. I apologized to him later, but I still feel bad about it.

Would I be wrong for throwing a snowball at my neighbor?
```

**Verdict:** KEEP

---

## 357 · `afafbf9160a5625c` · 998c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA For not paying a friend who bought me a drink?

```text
Bit of backstory, im 17, not legally able to buy my own drinks or even drink at all Every now and then a group of friends and I have a catch up, few drinks throughout the night, etc. 18yo friend wants to have a drink, so asks to stay at mine for the night, as he can't drive home after drinking, so I say yeah sure no problems. I also ask if he can pick up a cheapo bottle of something so I can have a bit to drink, as my parents didn't have the time to get me anything, and he says no, I don't follow up any further. FF to the party, he shows up and hands me a bottle of premium vodka, says I owe him the money for it. Throughout the night he's mixing stuff straight into the bottle, offering it around to people, acting like it's his own, and the next morning it gets tipped out due to the amount of crap in the bottle. He sent me a reminder that I owe him for the drink, but I don't think I do, and haven't said anything about it.

Would I be wrong for not paying a friend who bought me a drink?
```

**Verdict:** KEEP

---

## 358 · `b03a6bd9c6c39261` · 1195c · ask=title

> **Title** (not part of the prompt): WIBTA if I got a friend a Christmas gift?

```text
I'm not very good at social situations. 

One of my best friends is struggling a lot. (A very recent development.)

The Christmas season is coming up, and I can't think of a person more deserving of some kindness. She has been by my side through so much. 

I messaged her today and asked her what her preferred shopping spot is for groceries. She gave me the name, and asked why. I told her I was just getting my Christmas list together. 

She messaged back "Dont get me anything. I can't afford gifts this year." I told her not to worry about getting us anything in return, that she has been an amazing friend for the past fifteen years, and she has done a lot to support (emotionally) me through a really tough time, and we wanted to show her how much her friendship meant to me.

That was yesterday. She hasn't answered since. 

I don't want to make her angry, and I don't want her to feel disrespected, but I know she could use the help right now.

We wanted to get her a $100 grocery gift card, a $100 gas card, and some basic necessities that I know she needs. It's not a whole lot, but it's my hope that it will help her out a little.

Would I be wrong if I got a friend a Christmas gift?
```

**Verdict:** KEEP

---

## 359 · `b04faed39346455e` · 785c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I don't get my sister an engagement present?

```text
Hey,

I'm a student and I work casually, so I don't have much money. I will definitely get her a wedding present, but should I get her an engagement present too? 

She said she doesn't care about receiving engagement presents, but my mum and partner both think I should - I don't know if my sister is just saying she doesn't care because she knows I don't have much money and she's trying to spare me the stress. As far as I know there will be no engagement party, and there will be travel + accomodation costs associated with the wedding - not sure if those details make a difference.

Obviously not a major conundrum, I mostly just want peace of mind regardless of whether I get an engagement present or not.

Thanks!

Would I be wrong if I don't get my sister an engagement present?
```

**Verdict:** KEEP

---

## 360 · `b0ad9bdc2b3439a4` · 850c · ask=title

> **Title** (not part of the prompt): WIBTA for going off on my boss/father for now knowing english after living in the us for ~20 years.

```text
as the title says, boss barely knows a lick on english and decided to instead learn it, he surrounded himself with other native russian speakers.
  
   Now that we need more workers, he is expecting either russians, or english professionals so he doesn't have to talk with them.

He has already declined, and scared away 5 new workers because he couldn't communicate with them and got angry, and the latest one he went off at me.

I'm the head manager here, and only english speaker, i do just about everything that requires english.

I'ts really embarrassing since he is family and the way he acts at work, and to american customers makes us loose lots of business.

Just as an example almost everyone in our family knows more then him.

Would I be wrong for going off on my boss/father for now knowing english after living in the us for ~20 years.?
```

**Verdict:** KEEP

---

## 361 · `b139d44ff8d076a2` · 1208c · ask=body+title

> **Title** (not part of the prompt): WIBTA If I bought two bus tickets so I could sit by myself and told others they could not sit with me?

```text
I'm in college; over breaks I normally travel using Megabus. I'm fat (and working on it) but that makes it very uncomfortable for myself (and I'm sure the person I'm sitting next to) on these long rides. 

I began buying 2 seats and also reserving 2 seats side-by-side so I wouldn't have to deal with others asking since I then have an assigned seat. General seats aren't assigned and you sit wherever you want unless the driver honors my having two tickets. There are limited reserved seats. 

I was scheduled for a trip but they just emailed me saying my trip was cancelled and I'd have to reschedule. I have to make this trip this week so I plan to request a new trip leaving later in the week. My dilemma is that there are no reserved seats available on any trip, only general seating.

**Would I be an asshole if I bought two bus tickets and turned down others when they ask to sit next to me?** Would it be rude if I said something like "Sorry, but I paid for two seats" or say it's taken. Or just try to board first, dump all my crap then get off and wait til everyone else has boarded.

Would I be wrong if I bought two bus tickets so I could sit by myself and told others they could not sit with me?
```

**Verdict:** KEEP

---

## 362 · `b1443edaf961e52c` · 608c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I returned my Ikea furniture because I don't need it anymore?

```text
I moved into my 1 bedroom apartments on April 1st 2018, and got all my furniture from IKEA on April 4th. I am moving to a smaller studio in a week, and have to downsize. IKEA has a 1 year return policy on everything. I don't need my dinner table & chairs, side table, dresser, and I also want to get rid of my current bed-frame because I just don't like it anymore. Everything is in good condition. I just don't want to go through the hassle of selling furniture at a loss when I could just return it and get everything back.

Would I be wrong if I returned my Ikea furniture because I don't need it anymore?
```

**Verdict:** KEEP

---

## 363 · `b15b688b49add9b7` · 1247c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told my personal trainer I didn't want to work out with older women?

```text
This is on mobile so please forgive formatting issues. A few months ago I went from a very active job to a desk job and put on a little weight. I haven't been really happy with this so a few weeks ago I started going to a gym and meeting with a personal trainer. It was going great until the gym changed its rules on how many clients a personal trainer had to take to train at this gym. From what I've heard it's hard for them to keep up and there is no shortage of people wanting to work with trainers, hence why they have become more demanding with numbers. The issue I'm having is I pay for one on one training with G. It's been going really well and I am losing weight but after the gym changed it's rules, she is double booking/triple booking her time slots. My time slot is a popular one and so for 3 sessions now I have had to work out with 2 older women (late 50's early 60's) who simply cannot keep up with someone who's 21. I feel like I'm losing momentum in my work out as I spend a lot of time standing around waiting for them. I have asked G to move me on without them but she refuses because she doesn't want to explain the same thing twice.

Would I be wrong if I told my personal trainer I didn't want to work out with older women?
```

**Verdict:** KEEP

---

## 364 · `b1af0da6209c5fdc` · 743c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I asked my friend for ALL my money back?

```text
I have a friend who I've loaned money multiple times, and he always pays back within the week. Nothing major, under a few hundred once. Usually only $50 or something. 

Anyway, he'll seem to pay me back in chunks, but lately he's been shorting me by stupid small amounts like $5. (example: I'll loan him $100, he'll pay me back $45 then $50 a few days later)

I haven't said anything yet, because I don't want to feel like an asshole when it's only a few bucks among friends (literally the price of a beer on a night out, who cares right?). But I also feel I should be paid back 100%? I'm certainly not going to let it affect our friendship, but should I ask for my extra $5 back??

Would I be wrong if I asked my friend for ALL my money back?
```

**Verdict:** KEEP

---

## 365 · `b275810b16a7a819` · 1125c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I packed up my STBX-roommate's things?

```text
My soon to be ex-roommate is moving out by the end of the month. They have taken the bulk of their things to their new rental but they have left some odds and ends behind. I would like to put the things they left behind that I have found while reorganizing (things that were in cabinets or left behind in the bathroom) into a box and set it aside for them to get. Eass of access for them and it gets their items out of the way for me.

Here is where I need to know if IWBTA. We have not been getting along and things are tense between us right now. Mostly due to a lack of communication on their part but also due to me being unable to keep my frustration to myself. They cannot commit to plans they make and as much as my obvious frustration is adding to their stress, their flaky behavior is adding to mine. We are both rather done with each other. 

Would doing this look too much like a passive aggressive attempt to tell them to get their shit and leave based on the tension? Or would they be an asshole if they were angry at me for putting their things aside?

Would I be wrong if I packed up my STBX-roommate's things?
```

**Verdict:** KEEP

---

## 366 · `b28219644d580568` · 546c · ask=title

> **Title** (not part of the prompt): WIBTA if I installed sprinklers in my yard for the purpose of spraying people who walk through it?

```text
I live on a corner lot, and people walk through my yard all the time.  I don't think I'm your typical cranky old man, but a lot of people really push my boundaries. I don't get upset when people step into the yard when a car is coming or something, but some people are walking 10+ feet into my yard like it's a sidewalk, they get close enough that they can easily see into the windows.  I occasionally find dog poop in my yard as well.

Would I be wrong if I installed sprinklers in my yard for the purpose of spraying people who walk through it?
```

**Verdict:** KEEP

---

## 367 · `b2d7ff4f4da11aa2` · 698c · ask=body+title

> **Title** (not part of the prompt): WIBTA if i refuse to sign up for DoorDash?

```text
So, my sister owes me a somewhat large sum of money due to not paying her phone bill (something that falls back on myself as I'm the one who offered to put her on my plan). She hasn't held a steady 9-5 job in years, and has resorted to jobs like Uber, Instacart, DoorDash.. Shes drives a ton and has been able to earn a sizeable bonus around $1500. She mentioned that if I were to create an account for DD, she'd drive around under my name to earn another bonus in order to pay off her phone bill. I'm just not comfortable with her doing this due to not knowing the TOS, and I don't know how the service works taxes wise. So, Am I The Asshole?

Would I be wrong if i refuse to sign up for DoorDash?
```

**Verdict:** KEEP

---

## 368 · `b499fa9f3f0d5842` · 1539c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for asking my friend to replace an item she broke a year ago?

```text
I have an 11x9 ceramic baking dish/casserole tray that I use fairly often when I cook. It's my only deeper bakeware (I have several baking sheets but this one is about 4 inches deep) so it's good for things like brownies or baked mac n cheese. About a year ago, my roommate at the time used it to cook something herself. This is fine, we had an agreement where we shared the different kitchenware we had brought, but everyone knew what items belonged to them. At some point while she was using it, my roommate did something that created a few large but thin cracks throughout the tray. She told me this straight up, and I told her it was okay, but asked her if she would replace it. She reluctantly agreed, but she said she didn't have the money right at that moment. I said that was fine, she could Venmo me any time. Needless to say, she never paid me back. The semester ended, then she studied abroad for spring semester, and this year we don't live together anymore, and I rarely see her/talk to her, although we are still friendly. I didn't bother to hound her for the cash. The tray still worked anyway, since the cracks were hairline and I didn't use it all the time. However, the cracks are starting to get noticably bigger, and I would like to replace it. It'll probably only be $15-20 to replace, but I'm a college student who's tight on money, and while I can afford it, I wish she had just paid me back or got me a new one when it initially broke.

Would I be wrong for asking my friend to replace an item she broke a year ago?
```

**Verdict:** KEEP

---

## 369 · `b4fb2a0c582a2a79` · 531c · ask=closer · scrubbed

> **Title** (not part of the prompt): WIBTA - new chapter in life

```text
Don't wanna turn this post into a overly long essay on the situation as there isnt much details that could be added let alone rellevent. In april i will be leaving for boot camp, as this will be a big change from the life I used to and currently live. I've been considering cutting off friends and family as I want to have a fresh start. In a way I feel I can't truly clean my slate if I still have remnants of the current and past me, leaving me not able to fully change who I am.

I'm not sure what to do here. What do you think?
```

**Verdict:** KEEP

---

## 370 · `b60fc99dd53680c9` · 979c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked my mother to celebrate mother's day with her the day after this year?

```text
I have a trip planned to LA for a friends wedding on May 11. It's a brunch so I'm leaving the night of to fly to SF where she has a work trip and is choosing to stay a bit longer so the 2 of us can hang out a bit, until the 15th. The one friend I plan on seeing works a stressful job full time, he and I like to rock climb together, so I know that we can climb outside (significantly preferred option) if it's the weekend, and only indoors if its a weekday night. It's Mother's Day that Sunday. A few details:  I've never been a huge holiday celebrator in general, and if I had to go with my gut I don't think she minds that much either, to me I see no difference in celebrating the day(s) after. If she told me she wants to spend Sunday with me explicitly, I won't question it. She is also helping me get to SF, and seeing her is the reason I'm stopping by SF in the first place.

Would I be wrong if I asked my mother to celebrate mother's day with her the day after this year?
```

**Verdict:** KEEP

---

## 371 · `b73a66fd415d005d` · 533c · ask=title

> **Title** (not part of the prompt): WIBTA if I uninvite someone to dinner because she is a vegan?

```text
I've been talking to this cute girl for a couple weeks and yesterday I invited her to a date, eat something then go to a movie. Today I asked her what she thought about salmon and she said "eewww I would never, ever eat fish". Don't get me wrong, my best friend is vegan, but I don't think things would work with someone that is. 

Ps: Please don't be rude, I am just trying to make the best decision.
Ps2: Sorry for typos, English is not my primary language.

Would I be wrong if I uninvite someone to dinner because she is a vegan?
```

**Verdict:** KEEP

---

## 372 · `b7414671d6dfc4fc` · 670c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I repo'd a laptop from my friend?

```text
About 5 months back, I sold my laptop to a friend. It had started collecting dust in the corner when I started working in an office so I figured that I might as well sell it and get some cash out of it. She was tight on cash at the time and asked if she could pay it in installments, which I didn't mind. It was $20/month until she paid off the total of $100. Not that much, right? (another friend said that he would've paid for more it, but that's not part of this discussion)  Fast forward to today and She still hasn't paid me in full for it. If anything, she seems to have settled for not paying me back at all.

Would I be wrong if I repo'd a laptop from my friend?
```

**Verdict:** KEEP

---

## 373 · `b7d72ed0e8bb3c17` · 411c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I return something I damaged to its shelf?

```text
I like to work by the policy that if u damage something before buying it you should take responsibility and buy it (unless you weren't planning on buying it in the first place). Many people tell me that it's ok to return it and take a different one but I feel it's the wrong thing to do. All opinions are welcome and I am not offended by anything.

Would I be wrong if I return something I damaged to its shelf?
```

**Verdict:** KEEP

---

## 374 · `b8740347fe2941d2` · 1251c · ask=title

> **Title** (not part of the prompt): WIBTA if I, one of the bridesmaid, tell the bride she should be involved in planning her bridal shower?

```text
My friend was set to get married on may and a week before she was going to have a bridal shower organized by all 4 bridesmaids but because of x and y conflict she will be eloping. 
 The bridal shower was going to be at most 10 people including bridesmaids. It was going to be one of those lingerie parties paid for and organized by the bridesmaids.
Now because she will be eloping she still wants a bridal shower but with the groom and double the people (25)

At this point I don't feel like it's a bridal shower but a pre-wedding party as she called it at first and at which point the responsability of it spreads to not just the bridesmaids, but her as well

Shes still referring it to a bridal shower but co-ed. 
Dont get me wrong, I'm not one to think that pink only belongs to girls and blue to boys, but a co-ed bridal shower just sounds like a party. Ive just googled co-ed showers and none have a shower feel to them, more like just a party. 

So basically, she's not going to worry about organizing or paying anything since she's not doing a ceremony but still wants the bridesmaid to do a bridal shower for her and 25 people.

Would I be wrong if I, one of the bridesmaid, tell the bride she should be involved in planning her bridal shower?
```

**Verdict:** KEEP

---

## 375 · `b8b1a2916200931d` · 1299c · ask=title

> **Title** (not part of the prompt): WIBTA if I bought a purebred puppy from a reputable breeder instead of adopting from a shelter?

```text
So I’ve been wanting to get my own dog for a while now. I want a dog that will be just my dog, not a family dog so I can take it with me when I move out of my parents house after finishing college. I’m 20, have volunteered hundreds of hours at animals shelters and have worked with dogs since I was 16. I’ve also grown up with dogs my entire life. I want to buy a purebred, AKC registered pup as my first dog. I want that because there’s a specific breed that is well suited to my life style. I also want to know the dog’s entire history, and can be present in my dog’s life from the very beginning. 

My family has frequently mentioned that I’m selfish for buying a dog instead of adopting. They say I’m encouraging puppy mills (which isn’t where I’m getting the dog from). They tell our currents dogs that I’m “shopping for a shiny new replacement dog”. It’s starting to get to me so I’m just curious if this makes me an asshole. I fully understand that there’s tons of dogs that need a home, but for my first dog I want to eliminate as many unknown factors as possible. When I feel more equipped to handle a shelter dog with many gaps in their history, then of course I would adopt a dog.

Would I be wrong if I bought a purebred puppy from a reputable breeder instead of adopting from a shelter?
```

**Verdict:** KEEP

---

## 376 · `b8fff8772b3a4af2` · 1466c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I threw a pile of tree limbs into my neighbors yard?

```text
To give some background, the houses in my neighborhood are spaced a bit apart, kind of like a normal suburb type feel. My house and the neighbor in question both have gigantic pine trees in our yards, I have two and they have about 4. These things are seriously monstrous and the pine needles are a nightmare. So, a few weeks ago, we had some pretty bad weather: primary harsh winds and pelting rain. Because of this, many smaller limbs from the pines trees came down and littered pretty much everywhere. Since I've lived here for a while, it was something I was used to, I picked up the limbs from the yard and roof, whatever. Well a few days ago, I noticed that there was a small pile of limbs dumped onto the corner of my front yard closest to this neighbors house. Ever since, I have been getting more and more aggitated by this. I don't tend to let a lot bother me, but I feel like this was just a stupid slap in the face. He would have had to go around his front yard, pick up all of the branches, decided they were from my pine tree and dump them in my yard. I've left them there and I cant help but just want to dump them on his property. Part of me wants to dump them in his backyard, another wants to toss them all over his driveway and the last wants to just straight up dump them in front of his front door. Or should I just clean them up and secretly flip him off when I see him?

Would I be wrong if I threw a pile of tree limbs into my neighbors yard?
```

**Verdict:** KEEP

---

## 377 · `b907333565dced84` · 2175c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for continuing to go after an achievement/mount after being almost begged not to?

```text
So my best mate and I both play World of Warcraft. I've never been level enough to do raids and things but with BFA I got that boost and have been running current content until recently. He introduced me to Mount Grinding and I'm having a good time. One that I'm running a lot right now, mostly to spite another friend who's been trying to get Invincible's reins since the expansion was first released is Icecrown Citadel. This is where it all started. I'm a completionist. I've been working on the Loremaster achievements for a while just to feel complete. Like I've achieved something and that everything is complete. He introduced me to the "Glory to the Raider" achievements and when I was first raiding brought up the "Glory to the Icecrown Raider" mount and said it's another I can go after. Fast forward to last night, he was getting upset and saying I can do anything BUT "Glory for the Icecrown Raider" he said he's 'unique' in his group and the undead dragon is aestheticly perfect for his deathknight and if I get it then he's going to be pissed and have to find a new mount. Now Am I the asshole if I still do the raids and get Glory of the Icecrown Raider? I'm not doing it primarily for the mount but for the achievement and having done it. The mount is secondary to me. We also play different factions(I play alliance, he plays horde) so we never play with each other and I don't know if I'm even considered a part of his friend group that he wants to be special in. Personally I think his argument is a bit childish and stupid to get mad at me for doing it too. If I do do it I don't plan on using the mount just so he can still be 'special' and feel 'unique' and because like I said. As cool as the mount is, it's secondary to me just wanting to complete things. Am I the asshole if I still do it even after he practically begged to me last night not to do it so he's still 'special'? TLDR: Boyfriend wants to stay 'special' in his group with his raid mount. Said he'll get pissed if I run the raids to get the achievement since I'll get the mount out of it.

Would I be wrong for continuing to go after an achievement/mount after being almost begged not to?
```

**Verdict:** KEEP

---

## 378 · `b93aa17699eebfac` · 533c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I held my friend's stuff until he paid me back (READ BEFORE YOUR DOWNVOTE)

```text
TL;DR at the bottom So, I live in close proximity to a comic shop and I and my friends love comics, they live far away so they tell their order, give me the money sometimes before or after, I have a friend who owes me about 39.75$, which is a decent amount at our age. My other two friends think it's a good idea because he owes them as well. TL;DR Friend owes me money, so I consider holding is 50$ and 50$ order until he pays me back.

Would I be wrong if I held my friend's stuff until he paid me back (READ BEFORE YOUR DOWNVOTE)?
```

**Verdict:** KEEP

---

## 379 · `babd9e7a81b1960a` · 1234c · ask=title

> **Title** (not part of the prompt): WIBTA if I keep a parcel that was actually refunded?

```text
About a month ago I bought something on eBay (for $170) I waited 2 weeks and the company still had not sent it. I messaged them numerous times without a reply, finally I got replies but with different stories saying yes it’s sent but due to a system error there is no tracking, then it’s waiting to be sent, and another not yet sent will be sent ASAP they can refund if it’s been too long. I had enough and was thinking it was a scam and so I asked for refund- this was almost 2 weeks ago. In the last 2 weeks I had bought and received the item needed from another store. 

Yesterday low and behold the original refunded eBay item has been delivered to me. I checked the invoice slip and it was sent about a week after the refund. WAITA if I keep it? If they chase me up I’m happy for them to send me a new invoice to pay but my plan is to not reach out to them first. It’s actually from huge company in my country (but their eBay name wasn’t under the company name which is strange because I’d bought from the company’s proper eBay previously) and it is technically their mistake. I do feel shitty about it but also didn’t intend for this and they can afford the loss.

Would I be wrong if I keep a parcel that was actually refunded?
```

**Verdict:** KEEP

---

## 380 · `bac742e52fde6c3c` · 1295c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I asked my housemate to clean up her dog's mess?

```text
TL;DR on bottom

I'm going to do my best to tell this story as impartially as possible. Some background info: I live in an apartment and share a room with two other dudes (one of which I'll call "M"), and the apartment is occupied by only one other housemate aside from us. For her sake, I'll call her "Q." Q has a dog that we'll occasionally take out to use the bathroom, let into our room to watch, feed/give water. One time when we were watching the dog in our room together, the dog darted into our trash can, took a bunch of tissue paper, and ripped it up under my bed (making a pretty massive mess). Ever since, we've relocated the trash can to the bathroom. I asked Q what I should do about it since it's her dog, and she told me that since M has vacuum duty, he would be the one to clean it up. Well yesterday, M was doing vacuum duty, but did not vacuum the dog's mess because it was "too big to vacuum."

What do I do here? Do I just suck it up and clean it for the sake of not being an asshole? Do I ask Q to clean it up since it's her job?

TL;DR: Housemate's dog ripped up a bunch of trash under bed, housemate told me roommate would vacuum it, roommate told me it was too big to vacuum, the mess is still there.

Would I be wrong if I asked my housemate to clean up her dog's mess?
```

**Verdict:** KEEP

---

## 381 · `bb0a4716a8738b4e` · 1720c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I don't get my roommate anything for his birthday?

```text
To start out I currently share an apartment with two other guys while going to school. We got stuck together in the dorms last year and decided to move out together. Last year we did birthday gifts for the other two guys and a third who lived with us at the time. I spent \~$30 or so on each birthday. Their birthdays were all during the school year so before we moved to where we are currently. My birthday was in June right in between moving out of the dorms and moving into our apartment so we weren't together for it. The original plan for my birthday was I was going to buy something nice online and they were going to buy needed accessories for it but I later decided to not buy it and to save my money instead.After moving into our apartment I asked them if they would be okay with us going to the store and buying a cheaper item would they be willing to just reimburse me for that. That way it would be nearly the same as the other birthdays, I just would've been able to choose what item I wanted instead of them choosing it for me. He flatly said he didn't want to come and that he did not want to do that because he didn't want to just venmo me money for my birthday or what not. I was annoyed at this point told him to do what he wanted to do and left with my other roommate to the store. I ended up picking up the item that I wanted and my other roommate helped me buy it as my birthday gift. We returned home and it hasn't been mentioned since. It hurt my feelings for a while and we weren't friends for a bit after that but now we hang out and get along fairly well. TL;DR. Got roommate a gift for his last birthday, he skipped mine.

Would I be wrong if I don't get my roommate anything for his birthday?
```

**Verdict:** KEEP

---

## 382 · `bbce6ba28559a051` · 1417c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I refused to some possessions at home?

```text
When I was 12, I started listening to classical orchestral music and I loved it. A couple months after I turned thirteen, I started working some neighborhood jobs so that I could afford to buy (almost) every instrument of the orchestra and maybe find some online courses to learn. I want(ed) to be a composer for video games and movies, and orchestral music is good for those. I thought it would be a good idea to record them for approval, so I would want the instruments and the skill to play them for that purpose. I completed my collection around my 16th birthday. I found some courses online that I paid for with leftover money and took a year to learn all the instruments. When I was satisfied with my ability to play these instruments, I recorded some music I composed while I was still collecting. I turned 18 recently and my adoptive parents want me to leave my instruments at home for their biological kids (who, by the way, have NO interest in playing an instrument. They've told me before. They've said stuff like, "I think its really cool that you can create music and play all those instruments, but that's not really my thing even though mom and dad want me to play an instrument."). They want me to leave them so that the kids will have "a wide variety" to choose from. I don't want to leave the instruments, and I want to tell them I refuse.

Would I be wrong if I refused to some possessions at home?
```

**Verdict:** KEEP

---

## 383 · `bc2e9548ded6801d` · 741c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I bought two seats on Southwest and wouldn’t give up one despite the open seating?

```text
I’m a nanny, and I fly with babies fairly often (toddlers now). One of things Boss and I did was buy an extra seat even when the kids were flying on our laps so that one/both of us could have a buffer between ourselves and other people. On a flight like Southwest, where seating is open, would it make you the asshole to refuse to give up that open “free” seat even if the flight was extremely full or two people wanted to sit together? There are lots of reasons for having that seat (it makes feeding easier, your diaper bag can sit there within easy reach, it’s a place to put the baby) but it’s not necessary for a comfortable flight.

Would I be wrong if I bought two seats on Southwest and wouldn’t give up one despite the open seating?
```

**Verdict:** KEEP

---

## 384 · `bcdc1d8a9e686cea` · 522c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked my kids which parent they loved more?

```text
This is more hypothetical than anything because I’m curious about what y’all think about this. I know a lot of parents that do this and I can’t help but feel it’s potentially developmentally or emotionally damaging to children when they’re asked “Do you love mommy more or daddy more?” or even worse “If we were to separate, who would you wanna stay with?” OR EVEN WORSE (and yes, I have heard this) “If one of us were drowning, who would you save first?”

Would I be wrong if I asked my kids which parent they loved more?
```

**Verdict:** KEEP

---

## 385 · `bfc297919d85f009` · 1182c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked my roommate to clean the bathroom?

```text
I share an apartment with another young woman. I don't know her exact age, I believe she's probably 23. Just about every weekend she goes out and gets dead drunk with her friends, which I don't care about. She's an adult, she can make her own decisions. However, around 11pm last night she came in, went to the bathroom and went to bed. At 5am I woke up to use the bathroom and found she had thrown up on the bathmat, and hadn't cleaned it up. She had also thrown up in the toilet and hadn't flushed. Not wanting to leave fresh vomit to dry in the shared bathroom, I threw it on the washing machine, and did a load when I got up at about 6am. When I went out to get milk at the store, I found blood on the downstairs doorknob, which I was too grossed out to clean. This is the third time I've cleaned up her drunken vomit, and to be honest I'm getting tired of it. We're not good friends, I don't drink ever, and I'm not happy about being consigned to being the only person who cleans up her bodily fluid messes. Afaik she's never cleaned the bathroom, even when she has left obvious blood on and behind the sink taps.

Would I be wrong if I asked my roommate to clean the bathroom?
```

**Verdict:** KEEP

---

## 386 · `c0314884c8c52f5f` · 941c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told my coworker he stinks?

```text
So I am a new maintenance technician. A year of experience and progressing very quickly. Got a new job recently with a traveling company doing repair in industrial environments. At this new job there is a guy who has been doing this for a long time, but hea kinda dumb. He is a nice guy though. The problem is that he smells terribly like body odor. I mean you can smell him 6-7 ft away. On top of that he's a close talker and loves to look over my shoulder while I'm filling out paper work. I really can't even stand it anymore. Everyone else calls him stinky behind his back (I don't cuz that's fuckin rude) and I feel bad for the guy cuz like I said he's not a piece of shit. Also, most of us dont shower in the morning because by the time you leave you're covered head to toe in dirt. I mean I put deodorant on in the morning so that I'm not stinking up the plant when i start sweating.

Would I be wrong if I told my coworker he stinks?
```

**Verdict:** KEEP

---

## 387 · `c0997e916396024d` · 1013c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for telling my buddy I don't want his girl to come?

```text
Scenario: I've been married for many years and long-time-solo Buddy has been with Girl for a year or 2 (finally!). We used to hang out a lot together and equally as much with him third wheeling it. Ever since Girl came along, Buddy has been an absent buddy, we rarely see each other and when we do, Girl has to be there. We have common interests so I don't know what's going on there. Girl is ok in short doses but she's a bit hyperactive. In the interest of full disclosure, I probably resent girl a bit for taking my buddy but try not to show it. Problem: I scored tickets to an event out of town later this month and asked Buddy if he wanted to come (like the old days I thought). He mentions girl so I told him that Wifey wasn't keen and wouldn't be coming. Buddy wants to come but so does Girl and now she's excited about having a short holiday. He's phrased it as though I have a choice but they'll probably be offended if I exclude her.

Would I be wrong for telling my buddy I don't want his girl to come?
```

**Verdict:** KEEP

---

## 388 · `c132b3b6cf7ace43` · 905c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I keep a log of “rent" payments to my parents and ask them to sign?

```text
First and foremost, Happy holidays and Merry Christmas ya filthy animals. 

Now, I'm 21 and times have gotten a little rough for my family financially. I have agreed to pay them $700 for rent and money I owe each month. They have offered the entire upstairs apartment to me but I'd rather the room I have on our lower floor. 

My parents are great, honest people but I have this gut feeling that I should be recording my payments to them for some reason. I would like to whip up a little log that has the date, the amount paid, and both of our signatures. 

I feel like it is a responsible thing to do but things like this  could seriously offend them (especially my mother). In a way I feel as if it chips at the trust and honesty that we are trying to maintain. How should I go about this without being an asshole?

Would I be wrong if I keep a log of “rent" payments to my parents and ask them to sign?
```

**Verdict:** KEEP

---

## 389 · `c228823c11908f16` · 668c · ask=closer · scrubbed

> **Title** (not part of the prompt): WIBTA its snowing in my area (Seattle area) and manager is asking everyone to come in or they will be written up

```text
So I work in customer service. I just moved from a sunnier state that never got snow. Monday it snowed and about half the staff called out that day because they couldn't make it in. Snow is forecasted for 5 of the next 10 days. I just received a group text from my manager saying that we all need to come in at our scheduled time this weekend, because we know that snow is in the forecast and that's not an excuse for not finding adequate transportation. I have a manual V8 with balding tires and no experience driving in the snow. I know that my position is important and would require someone else to cover my shift.

I'm not sure what to do here. What do you think?
```

**Verdict:** KEEP

---

## 390 · `c2a7a53430f9f185` · 1296c · ask=body+title

> **Title** (not part of the prompt): WIBTA for complaining about my new neighbor's small child

```text
I am more than a little bit frustrated with my new neighbors. I live in a 4plex next to a big university. The area is mostly students, but my new neighbors have a small child. I don't really know anything about children but it walks and talks, but I don't think it goes to school, so whatever age that is? The neighbors are the unit next to mine so we share a front door entryway and a wall of our apartments line up. They seem nice, if maybe a bit weird. So far it's all been ok.

However, the kid makes a lot of high pitched screachy noises. It's so high pitched it could almost break glass and loud enough that I can hear it over my noise canceling headphones. My previous neighbors in that unit had a newborn baby and I literally never heard it cry, but I can hear this new kid several times a day. I called my mother and she said it's just normal happy child screaching and they just do that sometimes. I'm autistic and super sound sensitive as well as having migraine issue. I really miss the peace and quiet from my previous neighbors. 

Would I be an asshole if I went to my new neighbors and asked them to keep their kid from squeeling? Or do kids just make that noise and can't be stopped so I have to live with it?

Would I be wrong for complaining about my new neighbor's small child?
```

**Verdict:** KEEP

---

## 391 · `c2e21944c95412e6` · 1225c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I confronted my friend about feeling excluded?

```text
So I was walking home from school with one of my best friends last Friday(we'll call her A), and another one of our mutual friends, who both of us are pretty close to(we'll call her B) tagged along. As I later found out, both A and B are going to hang out at A's place after school. This is the part where I was wondering if they would also invite me along, seeing as they didn't have much reason not to.

So, we get to A's place as we're walking, and both of them are ready to go inside. However, as I'm about to just leave without mentioning anything, A turns to me and says, word for word, "hey, B's mom is also at my house talking to my mom, so I feel like that'd be wierd to have you over."

 So, being the peaceloving soul I am, I just kinda accepted it and went home myself. However, later I was thinking about the whole thing, and I realized it was kinda wierd, because if it were me instead, I would've just invited the person in too.
 
This isn't a parent I haven't met before, and I'm actually pretty close to her. I kind of want to talk to my friend and ask her what she meant by that, but I'm just not sure what to do here, am I being insecure?

Would I be wrong if I confronted my friend about feeling excluded?
```

**Verdict:** KEEP

---

## 392 · `c3b925f91195a0c2` · 640c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA to let a friend know that I don't remember the joke?

```text
Many years ago I was at a party with a group of friends and (not unusually,  we had quite a bit to drink). During the course of the night,  the husband of a very good friend told me a joke. The punchline was something along the lines of 'picking up a fucked up duck '. I remember that we howled with laughter and it was the best joke I'd ever heard in my life (or maybe it was just because I was hammered). Anyhow,  ever since then,  whenever we see each other, he says to me 'So Lissy, how's the duck going? ' and we both have a good old chuckle about it. I'm tempted.

Would I be wrong to let a friend know that I don't remember the joke?
```

**Verdict:** KEEP

---

## 393 · `c475222ddb6cea6e` · 537c · ask=body+title

> **Title** (not part of the prompt): WIBTA If I crash rent-free while they don’t live there?

```text
I am doing summer school at my college so I cannot go back to my home town. The place I live closes over the summer, but luckily I have some good friends that have an apartment. There are 4 of them, none of which will be living there this summer. I need to ask them to stay over the summer (which they will say yes to), but should I offer a portion of the rent? Keep in mind I am a (very) broke college student, but I don’t want to take advantage of them. Any thoughts?

Would I be wrong if I crash rent-free while they don’t live there?
```

**Verdict:** KEEP

---

## 394 · `c49b40910e48664a` · 417c · ask=title

> **Title** (not part of the prompt): WIBTA if I order pizza or food to be delivered to my house right now?

```text
I'm hungry and snowbound. I want to have something delivered but it's freezing cold outside, there's a foot of snow on the ground, and my house is located on a steep uphill U-shaped driveway (which means the deliveryperson will likely have to park at the bottom of one of the ends of the driveway and trudge all the way up to my door).

Would I be wrong if I order pizza or food to be delivered to my house right now?
```

**Verdict:** KEEP

---

## 395 · `c556688b013f5310` · 876c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I put someones stuff in a lost and found?

```text
Okay so some context real quick, this stuff is not lost. So in my schools locker room everyone is given a locker and a lock and told to put there stuff in their locker. As the year has gone by people have been losing their locks (somehow) and the whole "use your own locker" thing has been losing popularity. When I went in I noticed that someone had put everything in to MY locker and not locked it (I assume he lost his lock). I was gonna just gonna move his stuff out my locker and go about my regular process, but that can almost guarantee that his stuff would get stolen. I decided I would just steal a seemingly unused locker for the day and get their earlier to hopefully get there before whoever is doing it, but I might not be able to do this. TL;DR: Someone is putting their stuff in my assigned locker.

Would I be wrong if I put someones stuff in a lost and found?
```

**Verdict:** KEEP

---

## 396 · `c621414e55610379` · 1459c · ask=title

> **Title** (not part of the prompt): WIBTA if I told my roommate he needs to find a temporary home for his cat.

```text
So my roommate rescued a cat a few days ago ( where I live, it’s kinda cold as balls ) and decided to keep it. 

I have no problem with him rescuing the cat, but it apparently had fleas and he gave it a few baths.  My roommate has also yet to take it to the vet to get him check out ( you know, no more fleas, has all it’s shots, he’s fixed, etc..... )

I have no issue with him rescuing a cat or wanting to keep it.  The thing is he’s been constantly late on rent/utilities and he flat out told me he doesn’t have the money to take it to the vet.  My roommate is also going to visit his parents out of state so he’s trying to find somebody to watch it ( I can’t bc I work and will also be visiting family ).  I’m also concerned if the cat decides to not pee I the litter box/it spread fleas in his room.  My roommate is leaving in 2 months and I know for a fact he won’t have the money to get his room cleaned before my 2nd roommate moves in.

There is nothing in our agreement explicitly forbidding him having a pet and my apartment is cool with cats.  Plus I also kinda like the little fluff ball.  But I know he doesn’t have the income to support it and since the cat hasn’t been checked out, I want the cat to find a new temporary home until my roommate moves out.

I own the lease, so I could tell him to get rid of it, but i feel that would be an asshole move.....

Would I be wrong if I told my roommate he needs to find a temporary home for his cat.?
```

**Verdict:** KEEP

---

## 397 · `c680f022d6c1cdaa` · 614c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for asking about a friend of a girl who likes me?

```text
I have this friend in our friend group who is interested in me. She made it really obvious to the point where I felt kind of uncomfortable. So I decided to confront her about her and just let her definitively know that I'm not interested. She told me she wasn't interested which is fine. But I have sinking feeling she actually is still interested. Anyways, I recently noticed a girl on her IG story who was kind of cute. For the sake of argument/judgement, let's just assume that this girl is interested in me and it's not a delusion in my head.

Would I be wrong for asking about a friend of a girl who likes me?
```

**Verdict:** KEEP

---

## 398 · `c6d6ac17c74955ab` · 1109c · ask=title

> **Title** (not part of the prompt): WIBTA If I singled a coworker out for toxic behavior?

```text
I usually try to bring breakfast for my coworkers when I can (usually in the form of donuts or other pastries) and try to make it where there's two for each coworker. Lately I've had a few people approach me and thank me for the item (we'll say donut from now on to avoid confusion) but also ask why they only got one when everybody else got two. The first few times I've written it off as poor planning from me. Maybe I accidentally forgot that day. Then it got to the point where it happened more frequently. And I started to double-check the numbers eventually I asked to see the break room cameras to try to figure it out.  Once I was allowed to see I figured out it was a female coworker doing it. Usually she's really kind and considerate. But when I confronted her about it privately she denied it, and when  i told her I saw it on the cameras she got really rude. Would a asshole if tomorrow at work I made a announcement that I'd no longer be doing that because she is being inconsiderate ? I should mention it's her day off tomorrow.

Would I be wrong if I singled a coworker out for toxic behavior?
```

**Verdict:** KEEP

---

## 399 · `c6e95e02762f7fc4` · 1454c · ask=title

> **Title** (not part of the prompt): WIBTA for not switching to a smaller room for my roommates

```text
This isnt a particularly dramatic post, but one where I'd like other people's input.

So I've been living in a 3 bedroom apartment with only 1 other roommate (both of us 23M) and when we moved in I got the "master" bedroom which was downstairs and he got one of the 2  smaller bedrooms upstairs. The master bedroom is only bigger by about a dozen or so square feet, so nothing dramatic. The only thing that really sets it apart is its downstairs away from the main living space and away from the 3rd empty room.

He also has a girlfriend who lives with us rent free for the moment and she sleeps in his room with him.

Now we have added a 3rd roommate (21F) who will be moving in soon and she will get the room next to his. 

But now he and his girlfriend have essentially demanded that we switch rooms so they get the master bedroom. They argue that since a new person is across from them their privacy will be interrupted and that since theres 2 of them they should get the bigger room.

I'm obviously opposed to this and argued back that I should get the room because I originally found the place and took care of all the paperwork and general tasks to secure it, and I'm the one who pays the bills for the place (he still pays me for his half of the rent though, I just actually write and send out the checks).

So yea again, nothing too serious but would like input regardless.

Would I be wrong for not switching to a smaller room for my roommates?
```

**Verdict:** KEEP

---

## 400 · `c70ad91be14a4aad` · 955c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told someone at the gym about their smell?

```text
So I go to a relatively small office gym. One with all the necessary equipment but just one of each. The gym on average has between 2-6 people in it at any given time and any more would make it crowded. I workout almost every day after work but I will run into a guy whom I'll refer to as SS (smelly shoes) maybe once a week. I'm going to assume it's his shoes that smell as they are some super old converse. I've been going to the gym for years so I'm used to a bit of funk, but man does this guy's shoes ever stink. If you get within 10 feet it is like having your face in a hockey skate, so bad that I genuinely cannot focus if I'm in that area. The issue is there is no where to hide, the entire gym smells way worse within 10 mins of him entering the room. I have to cut my workouts short every time he shows up. I know it's going to be awkward but I don't want to be an ass about it.

Would I be wrong if I told someone at the gym about their smell?
```

**Verdict:** KEEP

---

## 401 · `c70f306d46f6c6a5` · 439c · ask=closer

> **Title** (not part of the prompt): Wibta

```text
There is this girl who has ruined my life for the last two years.He took over everything I liked because it was in her interest and I have no way to get rid of her and I finally cut her loose and she still makes me feel terrible randomly. Would itMake me an a hole to post a picture of her on our/roast me. No one else would see put me and it would be for my own enjoyment. Please vote so I know your opinions. Thank you

What should I do?
```

**Verdict:** KEEP

---

## 402 · `c8b7b95d48d744b8` · 1124c · ask=body+title

> **Title** (not part of the prompt): WIBTA for pursuing my feelings for my professor?

```text
I want to know if I'd be an asshole for this and ruin a professional relationship I have.

I'm falling in love with my professor.  I feel like if I pursue these feelings, I could potentially risk their career.  For nearly 2 years, I had have the strongest feelings for them.  I've never been so hung up on a person like this before.  We actively talk and we know each other very well.  I think they know that I have feelings for them, and I think they at least have some feelings for me, but they can't act upon them because they need to remain professional.  I respect that entirely.  

I just don't know what to do.  I want to with them or at least let them know how I feel, but I don't want to jeopardize our friendship and their career they worked so hard for.  Also, my professor is not married and is not in a relationship with anyone at the moment.  

I tried to move on and see other people in the past, but with other people, that emotional connection is not there, and my feelings for professor continue to persist.  

Would I be an asshole if I pursue?

Would I be wrong for pursuing my feelings for my professor?
```

**Verdict:** KEEP

---

## 403 · `c94bff45aab4de29` · 1025c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for wanting to play my PS4 offline at my friend's expense?

```text
Okay, I'll do my best to make this brief. I recently found out I will have to go to Wisconsin for some medical reasons unimportant to the story. For years, my friend and I have "game shared" on PS4. This allows only one copy of the game to purchased but it can be played on two consoles. So my friend and I, trying to save some cash, decided to buy a bunch of (digital) games through my account but split the price because his PS4 would get the game as well (gigantic mistake). The issue with "game sharing" is that the PS4 games we bought cannot be played offline on my PS4, but they can on his. With my trip coming up, I will be offline for a few weeks and would like to play my games during that time. If I were to un-game share (which I don't need his approval to do), all the games we split the payment on would be unplayable on his PS4, but playable on mine. We split the price of almost all the games, so he spent money on them as well as I.

Would I be wrong for wanting to play my PS4 offline at my friend's expense?
```

**Verdict:** KEEP

---

## 404 · `c9570a54a3ac8a5e` · 907c · ask=title

> **Title** (not part of the prompt): WIBTA if I bought a fur coat?

```text
For background, I am chronically cold all the time. It's to the point where I have been to several doctors to see if there is a medical issue, because if temperatures get lower than about 60 I feel cold down to my bones. My hands and feet stay cold for several hours even if I am inside and under blankets. My average apartment temperature stays at 77 unless I have people coming over, which is rare to be honest. Point is, I'm pretty miserable and with winter coming I know it gets worse. 

I've been through a lot of coats, from the puffy kind to the ones with the reflective interior that is supposed to keep heat in, and none have worked very well. I saw a fur coat at a thrift store and am seriously thinking about buying it, because my grandmother used to have one and would complain about sweltering in it. At this point, sweltering sounds pretty good to me.

Would I be wrong if I bought a fur coat?
```

**Verdict:** KEEP

---

## 405 · `c9801f59c39539eb` · 1326c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told my roommate to not let her boyfriend live with us anymore?

```text
So I live in a college house with a lot of girls, one of the girls has her boyfriend stay over every single night (basically live) at our house for 4 months. HE DOES NOT PAY RENT OR UTILITIES. He has his own room at another house a mile away and he does all of his showering there. His main impact on my life is the fact that he uses the kitchen a lot, takes a ton of time in the bathroom, and parks in our 2 spot driveway. I have tried to talk with my roommate multiple times about how I feel calmly and respectfully about what we could do to make our living situation work for everyone, whether that would be financially contributing or being added into the cleaning schedule. The last time I tried to talk to her about this she screamed at me that it’s just a personal problem I have with her and I’m a bitch. To be fair she has some valid points, I don’t really like her, because she does stuff like this and is generally disrespectful. I haven’t always been the best roommate, I’m kind of obsessive about cleaning and could do a better job of expressing myself. But I’m pretty sure that it’s pretty common to feel like people living at a house should contribute. Also if it matters it’s 2 against (me and pal)/3 don’t care/4 in favor.

Would I be wrong if I told my roommate to not let her boyfriend live with us anymore?
```

**Verdict:** KEEP

---

## 406 · `c99bceab4b4826c9` · 761c · ask=title

> **Title** (not part of the prompt): WIBTA if I tell my neighbours to move their garbage cans

```text
So, my neighbours always leave their trash cans beside the porch, in-between the space separating our homes. During the summer, they can kind of get smelly due to heat, but mostly they’re just an eyesore. I don’t have a problem with it, but my parents want to leave a note asking the neighbours to kindly move the bins into their garage/backyard/away from our porch. My mom especially is a really neat person, and the smell and sight of them just really bugs her. They’re wondering if a note would come off as passive aggressive, or a petty thing to do since it isn’t a HUGE deal. They definitely do not want to offend the neighbours and the garbage cans are technically not on our property.

Would I be wrong if I tell my neighbours to move their garbage cans?
```

**Verdict:** KEEP

---

## 407 · `c99f87a90854828b` · 795c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I didn’t wear a charity shirt given to me for a race.

```text
I’m running the NYC Half Marathon in a few weeks under a charity. The race has 20,000+ runners but sells out pretty quickly and once it’s sold out the other way to gain entry is through fundraising for a charity. I chose a charity I’m genuinely passionate about and hit my fundraising goal. The charity mailed out some shirts with the name of the charity and the charity team. While I really like the shirt, it’s a bit large for me and not something I’ve run enough in to know how it will feel in a race or effect me with things like chafing. I usually wear the same set of clothes for races because I’m accustomed to running in them and I’m trying to run a competitive time to potentially qualify for the NYC Marathon.

Would I be wrong if I didn’t wear a charity shirt given to me for a race.?
```

**Verdict:** KEEP

---

## 408 · `cbbdab20e7529efa` · 1102c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for refusing to pay my roommate for internet.

```text
When I moved in I set up my own router and modem. I cut the bill from $82 to $55 dollars and had to set up new internet because she was being uncooperative. She claimed that she didnt know who the internet was registered under, even though she was paying for it. So I told her to call comcast and send back the router when I set up my internet. She said she would and a few days later I texted her about it. She again said shed do it. Today she got a charge for $82 from comcast and is saying we (me and the other roommates) must help her pay for the internet. I sent her a screenshot of me telling her to send back the router and I have not yet flat out refused but I sure as fuck am not paying. Background, she is 27 and partially owns her own company. She always complains about how hard she works and never has time to do anything. This all happened during my week away from the oil patch. I work 120-168 hours per week. I was bombarded with messages. Lastly she does occasionally watch my dog but really my other roommate does that.

Would I be wrong for refusing to pay my roommate for internet.?
```

**Verdict:** KEEP

---

## 409 · `ccd0b167502085c7` · 1086c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I went out instead of spent time w my family?

```text
im on spring break right now and i just got over the flu after spending 4 days dead in my bed at school and then calling off my spring break plans and somehow making the flight home for spring break

i went out with my extended family sunday despite not feeling so good, but since im so behind on my work due to be sick and my mom works 9-5 M-F ive spent most of my time studying. i went out for 3 hours tuesday and thats it.

i leave monday and plan to spend all of sunday with my extended family. tomorrow im busy 9-3, and even tho my mom wants me to come out to my sisters baseball game tomorrow i want to stay home after my errands/doctors apts & do this project. then ill go out with them sat morning to the green market, go out sat night, and then hang out w fam sunday. i might also go out friday night instead and do the project sat

my mom always makes me feel so shitty when i go out like this towards the end of my break. would it make my the asshole if i went out anyways and only spent one day w my family?

Would I be wrong if I went out instead of spent time w my family?
```

**Verdict:** KEEP

---

## 410 · `cd78ec8b1c88daca` · 542c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I talk about a future job with a company I'm leaving

```text
This is the situation: I work for campany A. Company A sends me to company B as a consultant. I got a job offer from company C with a lot better pay for the same job (as a consultant). I took this offer and I'm giving my 4 weeks notice in a week. Now I really like working at B, the job and People are great. And my goal for the future is stopping as a consultant and working at a company and buy a house near the company. After leaving A ( and indirectly B)? Thanks!

Would I be wrong if I talk about a future job with a company I'm leaving?
```

**Verdict:** KEEP

---

## 411 · `cdfe35a95a89701d` · 1695c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for flipping on my roommate about the cat box?

```text
I'm pretty mad right now, and of course I wouldn't "flip" but I'm really tired of being nice about this. The cat box has become an issue between us. We live in a small apartment and it's in a closet off of our living room, so the smell can creep into our living room when not taken care of. She cleans it once every four days (scoops just the clumps and never replacing the litter entirely or scrubbing the box at all) and honestly our apartment just smells like cat litter and poop. I've had to become complacent because every time I explain to her nicely that I pay the rent just like her and would appreciate if she cleaned it every 1-2 days instead, especially since he's a big cat and lays huge turds. Every time I asked her, she would get kind of defensive and would tell me that I jump all over her every time she makes a mistake, and that she's only human. I get that she is only human, but this happens a lot. I just got back from a week of spring break. She left the day after me, and is coming home in about two hours. I come inside and it REEKS. I check the cat box room, and lo and behold, it's full of shit. She let shit fester in our house for a week. Took the cat with her, but left the litter. I don't know how to approach her about this. If anything, I am moving out in May. But I am just so, so livid. Especially after talking to her multiple times and it being known that the cat box is a source of serious contention between us. Or do I have to say it calmly, again, that her cat box cleaning habits are fucking disgusting? Or not say anything because I am moving soon? P.S. Love the cat tho. He's a good boy.

Would I be wrong for flipping on my roommate about the cat box?
```

**Verdict:** KEEP

---

## 412 · `ce027928c2f55abe` · 590c · ask=title

> **Title** (not part of the prompt): WIBTA for not tipping?

```text
The common defense of tipping is that it could go away if servers were paid a livable wage, but they're not, so we tip. In Minnesota (and other states, I'm sure), tipped positions still make minimum wage. Minneapolis, specifically, is on its way to a $15/hr minimum wage. The common defense of a $15/hr minimum wage is that all jobs deserve a livable wage. So, would I be an asshole for not tipping a server/delivery person/etc in Minneapolis? They're already making a livable wage, and significantly more than a server in other states(NE is under $3/hr).

Would I be wrong for not tipping?
```

**Verdict:** KEEP

---

## 413 · `cec4ae944faa4969` · 1199c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for returning dog to shelter

```text
I have been looking for a dog for about a month. This past weekend I saw a dog that was very friendly and cat safe. He seemed perfect. I was looking to foster a dog from the animal shelter, but they happened to have a sale for $15 dogs for that day only. In the rush of the moment I adopted a dog. I could see that he was a big dog, but he is morbidly obese so I thought it was mostly fat. When I was filling out the paperwork adopting him the animal shelter volunteer read his medical records and said he is 87 lbs currently but will be 60 lbs at a healthy weight. I felt like I had made a mistake almost immediately after leaving the shelter. I felt ready for the commitment of a dog, but after getting a dog I’m not so sure. I would like to move to a larger city and I am not sure that an 87 lbs (60 lbs assuming he loses his excess weight) dog will be feasible for apartment shopping in a new city. There is a 30 day return policy at the shelter and I’m torn with what to do now. TLDR; Really thought I was ready for a dog -adopted dog from shelter, but now I’m not sure that I am as ready for the life-long responsibility as I thought that I was.

Would I be wrong for returning dog to shelter?
```

**Verdict:** KEEP

---

## 414 · `cf8493c0782d039d` · 429c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I tell my mom what she needs to hear?

```text
My mother chose her boyfriend over me and my sister a couple months ago and left. Lately, she’s been drunk texting me saying she misses me and all that bs. My family tries to avoid hurting her feelings and I’m tired of it. She hurts everyone and thinks she’s in the  right because no one will tell her otherwise. They will on occasion, don’t get me wrong, but not often.

Would I be wrong if I tell my mom what she needs to hear?
```

**Verdict:** KEEP

---

## 415 · `cf9d030d48b60a6d` · 619c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for asking to get paid?

```text
So recently the I posted here that my father and I had a pretty significant fight. We have since talked a lot and come to compromises. The issue here is he now wants me to come work with him. I wouldn't mind, but I'm working a full 19 credit hour schedule, and I'm planning on working 30 paid hours a week. The goal is that if these compromises don't last, I'm not out on the street with nothing. The way I understand it, he basically wants me around as an unpaid intern to pick up some pretty basic skills. I would only be asking for minimum wage if I did go to work with him.

Would I be wrong for asking to get paid?
```

**Verdict:** KEEP

---

## 416 · `d00fdaf8154949cd` · 1194c · ask=body+title

> **Title** (not part of the prompt): WIBTA For texting my former roommates fiance about the roommate owing me money.

```text
So pretty much about over a year and a half ago my now former roommate came into my room and told me that his bank account had been hacked and he didn't have any money to pay rent. As I had already sent my half to him I was forced to pay the entire rent plus my half a second time. He promised me he'd pay me back and as we be friends for 5 years I had no issues with it. 

He started okay paying back a few hundred of what he owed before just stopping altogether. We got into a little argument about it as he tried to scam his way out of over half of it, so we came to the agreement I'd move out if he would pay me. I never forced him to pay me all at once or anything, I don't care if he gives me $20 a month I'd just like my money. 

Now that I'm moved out I saw that he'd gotten engaged to his long time gf that I'd known for a few years as well. When I sent him a text asking him if he'd forgotten about the money, it had been just about a year since I received anything, I got no response. I know his gf is a nice and reasonable woman. Is it okay to text her because I know she'll do  somethinf?

Would I be wrong for texting my former roommates fiance about the roommate owing me money.?
```

**Verdict:** KEEP

---

## 417 · `d03e80c58e64af7d` · 862c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I dont allow cousins to grow up together

```text
A little bit of background, my sister and her husband are both narcissists, we havent been close in years. Only catch up a few times a year in family situations, and usually try to keep the peace. Have so many stories about their behaviour, but that's for another time. She has recently given birth to their first child / grandchild for all families involved. My other siblings and I can already see how this poor kids life will turn out with 2 narcissistic parents, and would love to protect this child as much as possible. I'm due with my first child in a few months, and my sister has started going on about how good that the cousins will be so close in age, they'll be the best of friends etc. Our parenting styles will be quite different due to our personalities, and our different partners etc

Would I be wrong if I dont allow cousins to grow up together?
```

**Verdict:** KEEP

---

## 418 · `d16bbd5cc28ba08b` · 1126c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for pointing out their itinerary is full of tourist traps?

```text
I've been talking to someone from Indonesia who wants to have her dream trip to Europe this summer. I live in the Netherlands myself and that's one of the countries she wants to visit. She sent me her itinerary and I noticed it's filled with tourist traps. These places no local would ever visit because only stupid tourists go there. I'm not talking about the Colliseum of the canals of Amsterdam. I understand you need to see those things once in your life. I'm talking about these specific expensive novelty bars and lame attractions and destinations that locals know they suck. Without her asking for it? It's HER dream trip of course, and I wouldn't want her to enjoy her trip less by following my advice. Hell, tourist traps wouldn't exist if the majority of people wouldn't enjoy them for some reason. She's a bit of a bubbly girl who just wants to have a by the numbers Eurotrip, I guess. I have the feeling she would be perfectly happy doing the tourist trappy things. I don't want to be the jaded know-it-all. I feel like I'm being a snob.

Would I be wrong for pointing out their itinerary is full of tourist traps?
```

**Verdict:** KEEP

---

## 419 · `d1bdbc31c7c303bd` · 1118c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked a parent to take their sick kid out of the theatre

```text
My family and I went to see a movie today, and I sat on the edge of our section. Another family sat next to us, and one of their young kids, maybe 5 or 6, sat on the edge next to me. As soon as they sat down, the kid started coughing and sniffling snot. He'd have a coughing fit every four or five minutes, and would sniffle constantly in between them. And these weren't small, dainty sniffles, these sniffles sounded like he was doing all he could to hold back a deluge of mucus from pouring out his face. To be clear, I'm not mad at the kid for being sick, I'm mad at their parents for bringing a sick kid into a public and quiet place. I thought about asking the parents to take their kid out or going to theatre staff to complain, but ended up not saying anything and just walking out because I didn't want to be a dick or make a scene. So, would I have been the asshole if I had complained? Either to the parents themselves, or to the theatre staff? TLDR: Sick kid sat next to me in theatre and wouldn't stop coughing and sniffling.

Would I be wrong if I asked a parent to take their sick kid out of the theatre?
```

**Verdict:** KEEP

---

## 420 · `d1fa0cfb78ebf74e` · 2057c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I distanced myself from my friends due to them inviting someone I dislike?

```text
First of all: mobile, formatting, blah blah blah So basically, my closest friend group consists of me and six other people. We are all really comfortable with each other, and we’ve all been friends with each other for years with the exception of one of us. At the beginning of the year, a kid who I’ll name John came to school. From the first week of school, I could tell that John was a really mean person. But I chose to be a nice person and not make assumptions, and tried to befriend him. However my efforts were to no avail, as this guy just chose to completely dislike me, being super mean to me and saying completely incorrect trash talk in front of my closest friends. He spread false rumors, made fun of me in front of the girl that I was trying to impress, and genuinely didn’t want to let me make any effort to be friends with him. Fast forward a few months, and this stuff has been continuing throughout, and I have stopped attempting to be friends with him. Important info: my close friend group is well aware of my history with John. Some of them are friends with him, but rarely does he hang out with us. Recently, for the past few weeks, we have started a tradition called cereal Monday’s. We eat chipotle, walk around town, and for the final act, we would head to the nearby giant, all get those small bowls of cereal, and eat at a nearby coffee shop. I would end up chugging the milk at the end, and it was our fun little group thing. However, yesterday they decided to just invite him to our Cereal Monday’s despite being well aware of our history. Upon adding him to our group chat, I was up front about removing him and tried to argue with them, but they still decided to defend him, and now he is coming with us no matter what. I removed myself from the group chat and now they are all mad at me. I don’t know what to do. I am considering distancing myself temporarily, but I feel like that will be the wrong way to accomplish my end goal.

Would I be wrong if I distanced myself from my friends due to them inviting someone I dislike?
```

**Verdict:** KEEP

---

## 421 · `d243157a7738e8ea` · 1714c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I rated this delivery driver poorly?

```text
Okay so normally I'm not the "complain to the manager" type... but when I use apps and they ask me to rate things, I try to do so honestly. In this case I'm using an app which you order food for delivery, and afterward it asks about the food, the app experience and the delivery. It also automatically adds a 15 or 20% gratuity, and I always put 20%. If you had a negative experience, they actually offer you credit on the app or a refund, so unless I'm missing food or anything... I mean, I know how to use the app, so no issues = +1, I got the right food = +1, the driver delivered said food = +1, right? Right, well tonight I ordered chinese food. I live in an apartment building so the drivers always call, and I go downstairs and meet them in the lobby of the building. So tonight the driver calls and says he's downstairs. I go down (I'm in my slippers, it's not like I'm going outside?) and... he's not in the lobby. I go to the door and he's not even parked... he's sitting in the car in the middle of the road in front of the building. He gestures kind of impatient, hands me the food and 3 drinks with no cup holder, and as he does he says quite abruptly. "Please hurry, it's too cold and I have many orders."... I know I have a right to be annoyed. I'm cold too. That's why I ordered food instead of going outside (in coat and boots rather than slippers) to get my own. However, if I complain on the app (it's asking if I was pleased with the delivery itself: I'm not) I don't know how that will affect the driver. So... Is it even worth being irritated about? (The food was warm enough, and other than wet slippers and a slight chill, I'm fine)

Would I be wrong if I rated this delivery driver poorly?
```

**Verdict:** KEEP

---

## 422 · `d26624b71678ba0d` · 952c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I stole my money back?

```text
So this started over Christmas eve dinner, when I excused myself from the table to not scream out while friends and family discussed politics. When dinner was finally over, I went back downstairs to see people off, when an aunt told me that she left me $100 because she didn't know what to get me, and that she left the money with my mom because I wasn't there. Fast forwards to Christmas night, after waving off the final uncle, I asked my mom for the money. She told me that she already spent it on lunch, and her reasoning is that that is how Christmas works, that "we use the things that other gives us to make their experiences better." I thought that sounded like bullshit, and after consulting my friends, they agreed that it was bullshit and I should steal my money back. Now I would never take $100 dollars from my mom without permission, but in this case I feel like I completely deserve the money.

Would I be wrong if I stole my money back?
```

**Verdict:** KEEP

---

## 423 · `d2fd9d53ee72b8bf` · 1542c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I had some friends over that I knew my roommate has issues with?

```text
Bit of background: I had a pretty solid friend group throughout two years of college. We all got along fairly well within our core group, and though issues arose, the close proximity meant these things usually had to be dealt with amicably. 

Present day, nearly a year after graduation: I'm living with two of my former classmates. I get along with them both, though one more than the other. We'll call this one A, and the other roommate B. 

Recently, B's girlfriend got into a row with one of our friends (we'll call her C) who tends to be a pillar of our group, the one who is at every gathering. This happened a few months ago, and I'm not good at keeping up with drama, but I do know that B and his girlfriend blocked this friend on every platform. I don't know how much the situation has changed, but given that they probably haven't seen each other since, I don't think it's improved much.

Recently, me and A hung out with a bunch of our classmates (without B), and we had a blast. I suggested that people come over to our place for the next bash, as we haven't had the chance to show off our apartment yet. A was on board, as was most of our group, but C said that B wouldn't wanna be there with her, and would probably take off to his girlfriend's. I haven't asked B yet, because there are two outcomes I see: either he agrees and feels the need to leave his own apartment, or he says no, and that just leaves us all feeling resentful. Thoughts?

Would I be wrong if I had some friends over that I knew my roommate has issues with?
```

**Verdict:** KEEP

---

## 424 · `d30bfa774ab97882` · 1132c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for asking to find out who my secret santa is?

```text
Last month I joined a secret santa for artists, and everyone had multiple weeks to complete a piece for the person they were assigned to give art to. I was one of the many who published their gift on time, and the person I drew art for was very happy with their gift. It's now past the deadline for gifts, and I'm rather annoyed that the person who was supposed to give me my gift never did their part. The host has decided to give everyone an extension but-- it's  only 10 days, and now I feel like I'm going to get a rushed gift (as it usually is with extensions) with no effort put into it. Obviously they dont care to put effort into my gift if they didn't do it yet, when they had a month to do so. I spent 3+ weeks on my drawing. I'll most likely accept whatever I get for the sake of civility, but since it is past the deadline, I'd like to know who I'm actually waiting on so instead of having to go and check the group folder in the hopes my gift is finally uploaded I can just subscribe to them and see their uploads in my feed once/if it finally is posted.

Would I be wrong for asking to find out who my secret santa is?
```

**Verdict:** KEEP

---

## 425 · `d469fb33dc9bf0d4` · 700c · ask=title

> **Title** (not part of the prompt): Wibta if i want my disabled brother to live in a mental home

```text
I have a twin brother that has low functioning autism and major OCD. When I'm doing anything I hear my brother squealing eeeeeeeeeeee all day and night. We try to give him an iPad but he knees them in half as soon as he gets flustered, and he's always flustered. If everything is not perfectly place exactly where it needs to be he'll be screaming and starts angrily biting his hand and starts grabbing for someone's neck. If that's not bad enough my nephew just keeps provoking him causing him to get flustered and guess who gets attaked, me. Am I the a hole if I want him to live somewhere that knows how to take care of him

Would I be wrong if i want my disabled brother to live in a mental home?
```

**Verdict:** KEEP

---

## 426 · `d5ca6f86bdd1438f` · 878c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I got rid of my neighbor's dog?

```text
I live in a apartment off-campus in a college town that I rent from a local realty company (LRC). The apartments are a series of units in a building with a row of separate buildings along the road. My unit is right next to another apartment complex, and my neighbor's is in that other apartment complex (but not mine) that are all owned by LRC. I can frequently hear my neighbor's dog barking at random times all throughout the day, and I usually hear it about once a day when I am home. I have been mostly able to ignore it up till now, but he was barking fairly late last night (I have to go to bed early for 8am classes). And I've remembered LRC has a no pets rule, in fact on their website it says in all caps "NO PETS ALLOWED." I'm also finding piles of dog poop around the section of grass that separates our buildings.

Would I be wrong if I got rid of my neighbor's dog?
```

**Verdict:** KEEP

---

## 427 · `d5fa10239a382422` · 536c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for not bringing alcohol to a party if I will not be drinking some?

```text
I have been invited to a birthday party for tonight. For several reasons (party location, personnal health choices,...) I do not intend to drink alcohol tonight. We've all been asked to bring a bottle in so we can have fun. As I do not intend to drink alcohol from anyone, does it seems "unfair" if I don't bring alcohol in as long as I do not have a dingle drop of alcohol from other people? I'd still bring some softs in of course but nothing else.

Would I be wrong for not bringing alcohol to a party if I will not be drinking some?
```

**Verdict:** KEEP

---

## 428 · `d6492b2168288751` · 1270c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I refuse to leave the gym?

```text
TL;DR at bottom So I'm a big basketball person, and I decided recently that I'd try to get back into playing shape. I renewed my membership at the local sports center and today was my first day back. Now, the multi-purpose room, which is essentially just an indoor basketball courts with dumbbell racks and a badminton net you can set up, is open use from 6-8:30 am. My plan was get in as early as possible so I'd be able to play without bothering anyone/being bothered, for maybe an hour or so until I can go longer. Today at around 6:40 these 2 guys came in with badminton equipment and asked how much longer I'd be staying, saying there's a group that usually plays from 6-6:30 and maybe I should join them. I just shrugged and said something along the lines of " Oh maybe, I'll think about it", and then left. Now, the badminton setup takes up the whole room basically because of how much space you need to move around, and I feel like taking up a whole room for two people is kinda selfish. At least what I'm doing only takes up half and can involve 5 other people. TL;DR I go in early at a gym to play basketball, two guys ask me to leave before I want to so they can play badminton even though the room is free use.

Would I be wrong if I refuse to leave the gym?
```

**Verdict:** KEEP

---

## 429 · `d70850e18d5e39bd` · 539c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I didnt want to take theater on my classes selection sheet

```text
I am signing up for classes for next school year and I ask my dad what I should do. He says to do theater so I can memorize more stuff and be fluent in speaking. I said that I didnt want to do that because I am not a theater person and I also heard from a friend thats not what its about. He says I still need to sign up so I van vet things into my head. I said what would I use that skill for and he goes  to being religious matters which I will not discuss

Would I be wrong if I didnt want to take theater on my classes selection sheet?
```

**Verdict:** KEEP

---

## 430 · `d725d005d1d95c0b` · 1352c · ask=body+title

> **Title** (not part of the prompt): WIBTA for not having a flower girl or ring bearer?

```text
I know it’s our wedding and I can do whatever with it, but I never knew how many people would be offended that I don’t want their children in the wedding. Ever since I was a kid I dreamed of having two small dogs as the ring dog and flower dog in my wedding. I never wanted kids in the procession or at the wedding in general. As I’ve gotten older, I kind of wouldn’t care if older kids were there but absolutely no little children. I’ve been to too many wedding where babies and little kids make too much noise during the ceremony or completely ruin the first dance or knock over cake or put their dirty little fingers in it or the food. I agreed to pay for 2 babysitters for the children at our wedding in a different room of the venue so they could at least be there, but so many moms have asked me about who was gonna be the ring bearer/flower girl. I told them that no children will be in the wedding, and my friends 2 little dogs who I absolutely adore will be walked down with the rings and flowers instead. I guess that’s the wrong answer because most of the moms look at me like I slapped in the mouth. A lot of people said they couldn’t/wouldn’t come. Even my MoH was a little offended that I wouldn’t even choose her niece who’s not even 3 yrs yet. Am I really the asshole here?

Would I be wrong for not having a flower girl or ring bearer?
```

**Verdict:** KEEP

---

## 431 · `d7e112d65ea8a5fc` · 1241c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA If My Friend and I Arranged a Sleepover Without my Brother?

```text
For some context, my brother has been friends with someone for years before me, but this year I also became friends with him since we now go to the same school. He has talked to me about his friendship with my brother, and how he often feels like he isn't supportive. My brother says these little small things like "you're bad" or "you're fat" as a joke to people, but the problem is he doesn't stop when asked to. My brother doesn't know boundaries, and as a result, this friend of his considers me a closer friend than him. We have recently been talking about arranging a sleepover, and the plans only included me. He told me he was fine with my brother coming over too, since they are still friends no matter their differences, it's just he only wanted me to come over since my brother has been over multiple times by himself, even when him and I were friends. My real problem lies with my mom, since she believes that not inviting him would be exclusion, and wouldn't allow me to go. However, I don't know if that is really true. I have been really confused on if it would be asshole-y to not invite him even though he has gone over by himself multiple times.

Would I be wrong if my Friend and I Arranged a Sleepover Without my Brother?
```

**Verdict:** KEEP

---

## 432 · `d7f38032b72279e9` · 1700c · ask=title

> **Title** (not part of the prompt): WIBTA If I left for another country to chase my dreams and cut contact with my family

```text
My dream is to be in some sort of military service as a grunt/infantry my current plan is to a) join the FFL or b) join the USMC (yes I know there are many challenges I will have to face if I try either one of these options)

However my plan also involves cutting contact with my family as I have a strong dislike for almost all of everyone within my family due to

A) my extended family being a bunch of judgemental lazy assholes

B) my close family member who is insane and refuses to get help no matter how much I try to suggest it/convince her and almost every interaction devolves into a screaming match (I get that it’s not their fault but I can’t handle it anymore)

C) my other close family who just ignore my existence almost entirely (we do have okay interactions on the odd occasion)

The only people I enjoy hanging with are my mates, to me they’re more like family but we’re all kind off splitting off to do our own thing i.e uni, full time work ect ect so basically to me it feels like there’s there’s nothing for me here in my current situation as I’m currently just working a dead end job

I don’t plan on telling anyone my plans or when I’m leaving as I don’t want to have to deal with the shitstorm that’ll arise from it

There are a few members of my family who I really do care about though and those are the few people I really am sad about leaving behind but I hope to stay in contact with them after I leave 

P.s if this does come off as me seeking validation I apologise and that wasn’t my intention, I sincerely do want to know if you guys think this is a asshole move or not

Would I be wrong if I left for another country to chase my dreams and cut contact with my family?
```

**Verdict:** KEEP

---

## 433 · `d81920eb06036786` · 1811c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I told My Friend She Was Being Childish

```text
Our prom happened on my friend's birthday. Let's call her Nic. Nic's mom was PTA pres and so could choose when prom happened (with a range of late Feb-early March). March 1st was chosen for 1.) being a Friday 2.) being the only Friday without another school event happening at the same time 3.) being her kid's birthday so she could save a bit before the big 18th birthday debut next year. Side note: they were very, very, ***very*** rich enough to not *need* to save, but extra bucks is always good, right? During the prom program, Nic is upset and all pouty. She talks to no one but her boyfriend, and only after the prom at a private party at a small bar at the hotel we were staying at (mine and Nic's friend circle + her mom and boyfriend) does he relay the message to us friends to find a way to leave the place because Nic was Not Happy Being There because she didn't have her own birthday party or even a cake or gift, and she felt that a small private party instead of a grand celebration like she had every other year but this 2019 "demeaned" her birthday. We friends hold in rolling our eyes. We know Nic has high standards and gift-giving is hard because 1.) she's rich 2.) she likes material gifts 3.) she's picky about what she receives and she *will* say it to your face if she hates your gift (Side note 2: this is the one year we didn't give her a gift because prom preparations. most of our circle is also not rich, so prom alone is a big hole in our wallets), but because birthday we restrain ourselves and pull her away from the bar in hopes she'd be happy. She wasn't. Pouting and stomping the whole time all the way up to our room. TLDR: Rich friend upset by not having bday party/gifts. I want to call her out on it next week.

Would I be wrong if I told My Friend She Was Being Childish?
```

**Verdict:** KEEP

---

## 434 · `d8bf2eebbb267139` · 1877c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I got angry at my friend for spending more time with his long distance gf?

```text
I'm friends with 2 guys, we'll call them Andy and Marco. We're all currently 18. Us 3 have been friends with each other since we were very young, and we've always thought of each other as our best friends. This has been changing recently. Andy met a girl online who lives in another country, and they fell in love. She lives very far from where we are, and I'm doubtful that they'll ever get to meet in real life. I'm very glad he's found someone that makes him happy, except he basically puts all of his time into her and spends much less time with us. He would pretty much ditch me and Marco. As an example, we'd ask him if he wants to get in a call and play a game, and he would decline saying he's not in the mood because "he wants to relax." Minutes later he would start a game with his girlfriend "because she asked." This kind of behaviour has gotten on our nerves a lot, and we feel both very hurt and like he's being a bad friend. We even sat down with him and talked about this before and he acknowledged everything and apologized. Well, that apology meant nothing apparently because he hasn't really changed at all. On one hand I can understand that he has very strong feelings and is infatuated with this girl, and I can understand wanting to spend as much time as possible with someone who lives so far away, but I don't think having strong feelings serves as an excuse for disregarding your friends. It pretty much feels like a giant slap in the face and a big "fuck you" to me and Marco that he rather ignore us and spend this much time with someone he's never met before over his lifelong friends, even after previously apologizing and saying he'll change. But on the other hand I want to stay understanding and sympathetic to him as he clearly has strong feelings.

Would I be wrong if I got angry at my friend for spending more time with his long distance gf?
```

**Verdict:** KEEP

---

## 435 · `d99cf7a7f50f61ac` · 407c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I report restaurant staff for refusing to serve me 5 minutes before their last order?

```text
Basically the title. The restaurant is in a hotel and they’re supposed to close at 1:00AM, last order is at 12:00AM. I get there at 11:55PM and am told that they are closed. I assume I get the hours mixed up. I get back to my room, and turns out They do close at 1. Would it be wrong if I complained?

Would I be wrong if I report restaurant staff for refusing to serve me 5 minutes before their last order?
```

**Verdict:** KEEP

---

## 436 · `d9df5bc59ed11dc6` · 669c · ask=body+title

> **Title** (not part of the prompt): WIBTA for going to a paid gig instead of volunteering like I promised I would do?

```text
So I have a very good friend that wanted a group of us to volunteer for her favorite charity as a present to her for her birthday. I said yes but completely forgot the day which she told me she signed us up for. She asked about a month ago. A few weeks ago, I got approached to work for that same evening for a catering company. It’s a ton of money for super easy work. I said yes to that, because I did forgot that we were supposed to volunteer. I don’t want to volunteer and need to the money right now. Am I an asshole for telling my friend that I don’t want to volunteer?

Would I be wrong for going to a paid gig instead of volunteering like I promised I would do?
```

**Verdict:** KEEP

---

## 437 · `dab80ff7fe2f332a` · 1028c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA If I requested a driving instructor that ONLY speaks english?

```text
So I've done all my hours and am doing my test in april, I've set aside money so I can have a driving instructor do a lesson for me every day for a week leading up to the test so I'm as prepared as I possibly can be. I've had one lesson from an instructor and he had a really thick indian accent, and I'm sh*t at understanding accents. The whole lesson I was focusing more on trying to figure out what he was saying rather than actually focusing on the road and driving, which made me feel really unsafe. I didn't learn anything from the lesson (a waste of $60 too, which for a broke student is a hell of a lot of money) and am anxious about driving with another stranger, especially if they have an accent. Anyway, I was going to call a bunch of driving schools today and ask if I can specifically have an instructor that only speaks english, as I don't want a repeat of the last lesson and I don't want to waste my money. Is that an asshole move?

Would I be wrong if I requested a driving instructor that ONLY speaks english?
```

**Verdict:** KEEP

---

## 438 · `db1253ace7191ff0` · 1536c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA For Reporting A TV Noise Complaint?

```text
A few months ago, one of my (32M) neighbors Jim with whom I share a bedroom wall with got a nice new TV for his bedroom. Since then, he has it going almost every night all night, from 10-11 PM to 6-7 AM. It sounds like it's directly up against that wall we share, which has minimal noise insulation to begin with. I measured it with a sound meter app (which I realize isn't the end all be all in terms of accuracy) and it tends to run about 42-50 dB on my side of the wall. I'm a relatively light sleeper and I've been losing sleep because of it for weeks. To top it off he occasionally comes back from the bar with friends to party well past midnight and is blasting music when his roommate who pays almost all the rent is out of town, though admittedly that has slacked off as of late. I've also been woken up a few times to screaming matches between him and his girlfriend, or him and his roommate who is letting him live there. I really dislike confrontation with neighbors especially when I know he's at least verbally violent and am concerned what will happen if I escalate. I've asked multiple times for the bedroom TV to be off at reasonable night time hours (11 PM - 7 AM) or use headphones. The last time I tried to ask, he cussed me out from across the wall "I can't FUCKING listen to the FUCKING TV? JESUS FUCKING CHRIST" and then turned the TV up a little. I'm going batty, I'm a working adult who needs his sleep and I feel like I'm not making an unreasonable request.

Would I be wrong for reporting A TV Noise Complaint?
```

**Verdict:** KEEP

---

## 439 · `db483999fcf67057` · 741c · ask=title

> **Title** (not part of the prompt): WIBTA if i ditched a friend for my grandfathers birthday

```text
little back story. im in the military. a coworker and i had signed up for some volunteer work a month or so in advance to do at a school for test proctoring. my grandfather, living 14 hours away(driving) in a different state, is having his 80s birthday. i didnt know if my family was planning on throwing him a party or not and if so what the date was.

low-and-behold. the birthday date would make it impossible to make it back to the volunteer opportunity day. as my drive back would be difficult.

my coworker is mad at me for setting up the volunteer work for us and than wanting to back out for my grandfather. i tried to reschedule the volunteer thing with no luck.

Would I be wrong if i ditched a friend for my grandfathers birthday?
```

**Verdict:** KEEP

---

## 440 · `db73235464f85965` · 1118c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I quit a job I hadn’t started yet.

```text
I interviewed an was hired to be a bartender/manager at a bar that is being built. Now I was hired in the beginning of January under the assumption the bar would be open around Valentine’s Day. I went all of February without contact from the person that hired me until I reached out to them in the beginning of March, and was told it was behind(understandable) and I would be receiving a call the week of March 18th. Well last week came and went and yesterday morning I get a message to “report to <address> at 3 pm on Tuesday March 26th”. I replied asking a ballpark amount of time I would be there and explained I have my son and would need someone to watch them. I received no reply. I have been wrestling with replying now “thanks but no thanks” but I’m not usually that type of person. They have intentions of opening this weekend and I’m assuming do some sort of training this week(haven’t been told about anything besides “reporting” today.)that I won’t be present for anyway since I have a job and was given 23 hours notification that this was starting.

Would I be wrong if I quit a job I hadn’t started yet.?
```

**Verdict:** KEEP

---

## 441 · `dc1492c452054ea1` · 1177c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for refusing to pay a collection agency debt for damages to my old apartment caused by my roommates pets?

```text
So a little more than a year ago i moved into an apartment with some friends of mine under the impression that anyone of us that owned pets would take care of pet deposits and such. All 4 of us signed the lease (two roommates in particular owning pets, I did not own any), and we moved in. Maybe a couple weeks into living there, the discussion of the pet deposit came up and it turned out to be very expensive, and so none of the roommates that owned the pets paid their deposits. Over the course of about a year, the animals gradually destroyed the apartment, the carpet was chewed up, the window blinds were ruined, etc. When we all moved out about a month and a half ago, the two roommates owning pets allegedly cleaned the apartment and took care of the damages the pets caused. Yesterday, we all received letters from a collection agency stating that we owe over $2000 USD in damages. Now, the roommates owning pets are talking about disputing the debt, but since i was on the lease, im worried that i might be locked into paying part of the debt.

Would I be wrong for refusing to pay a collection agency debt for damages to my old apartment caused by my roommates pets?
```

**Verdict:** KEEP

---

## 442 · `dc2c61200fa45722` · 1594c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told my roommate/best friend I wanted to stop sharing food with her

```text
So I'm roommates with one of my best friends in the world. I love living with her 90% of the time but recently she has been getting on my nerves. We've always been very open about sharing things. It's usually not a problem for me - we only have so much cabinet space and we don't need 2 jars of every spice for example. As long as no one uses the last of my stuff or replaced it, I've normally been very chill about. She has never used my stuff excessively. Until this year. She has been exclusively using my stuff. And not just a little bit. I had a jug of olive oil from Costco and she probably used 75% of it without paying a dime for it. She uses my cheese all the time. I was really excited to eat see Mac and cheese only to find that she had helped herself to the best flavors in the variety pack. She doesn't even ask and just helps herself. It would be one thing if she helped pay for this or if I used the stuff she bought equally as frequently, however she only buys things that I don't like (and therefore wouldn't buy). We only have until May in this apartment (we'll be graduating college). I'm kinda at the end of my rope with this. I don't want to rock the boat and have a miserable last semester of college and be an asshole to one of my best friends by suddenly reversing my open sharing policy. But I also don't want to subsidize all hereals because I really don't have the money to do that to begin with. I don't have a meal plan in order to save money but she still has a meal plan.

Would I be wrong if I told my roommate/best friend I wanted to stop sharing food with her?
```

**Verdict:** KEEP

---

## 443 · `dc7633712e8d51e8` · 400c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I show up for free food and leave?

```text
Hey guys, standard broke college student here. So, the university offers free food for students that show up and listen to a presentation. Sometimes the presentation requires chatting in small groups / other participation, and it usually lasts an hour long. I don't have time to stay for the full session, but would stay if I didn't have class.

Would I be wrong if I show up for free food and leave?
```

**Verdict:** KEEP

---

## 444 · `dd37292b278910df` · 1114c · ask=title

> **Title** (not part of the prompt): WIBTA if I purposely ditch the parents im supposed to babysit for?

```text
Quick backstory I help babysit for a "friend" since I have free time. They have 3 kids and they are the sweetest sometimes. I've been doing this since August and known them since May. On occasion I would babysit for 2 hours and the parents would stay out for 3 hours. They wouldn't pay me for the extra hour but that didn't bother me at the time. Now it's coming to the point where they say they would be out for 3 hours and not come back for 4-5 hours. When I told them about it they told me they lost track of time. Their lateness has caused me to be late to appointments I had. I missed an important event ( football game lol )

What really got under my skin is when they make comments like "Oh carl you don't have kids or a job it's not like you're missing anything important" or "Carl you have plenty of free time, we hardly get any". They have an event this weekend and I really want to tell them I'm coming but never show up so they miss their event. I've talked to them about their lateness but they keep making snide comments.

Would I be wrong if I purposely ditch the parents im supposed to babysit for?
```

**Verdict:** KEEP

---

## 445 · `dd67c702c1f55024` · 901c · ask=title

> **Title** (not part of the prompt): WIBTA for kicking my friends/roommates sister off our Hulu account

```text
My roommate and I share a Hulu account that we each pay $6/month for.     My friend added a profile for his sister to use. She doesn’t contribute to the account because in my friends reasoning “she’s 14 and has no job and no money”. I wouldn’t have a problem but the thing is, Hulu only lets 2 videos play on an account at the same time. So if both my friend and his sister are watching stuff, that means I can’t watch anything even though I pay for the subscription. 

I don’t think it’s fair that someone who isn’t paying for the account is indirectly stopping me from enjoying something that I’m putting money into. I’ve tried talking to my roommate about it but he’s kinda dismissive and last night when I was having problems he seemingly ignored my texts about the issue. I don’t know 100% if that’s the case though.

Would I be wrong for kicking my friends/roommates sister off our Hulu account?
```

**Verdict:** KEEP

---

## 446 · `dd79d13a4cf69f6a` · 517c · ask=title

> **Title** (not part of the prompt): WIBTA if I asked her to be a homemaker?

```text
Gf and I moved in together in September.   She lost her job.  She’s been trying to find one since, but hasn’t had any luck.  

Personally I don’t believe that.  She was applying for minimum wage jobs.   She could have found one by now.  

She hasn’t.  When I come home the place is a mess.  When I want some food we have to order take out.  

I’m not a misogynist, but I believe in sharing the load. Since she’s not working I feel like she should take on this work.

Would I be wrong if I asked her to be a homemaker?
```

**Verdict:** KEEP

---

## 447 · `de58ddcd5457f1b1` · 1418c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for telling my roommate’s friend to shut up ?

```text
Ok so I live in a small apartment and have one other person I live with, we will call him Billy. He has a friend that I would say Im acquainted with at this point, we will call him Jake. Jake comes over at around 7 p.m. last night to record music with Billy. Billy is on his PlayStation playing a video game so Jake starts talking to me when I already say I have to do stuff around the apartment. I’m ok with it at first because I’m in the laundry room doing laundry, but then I’m in my room. Jake decides to follow me to my room and continue to talk to me while I’m cleaning up and this is where I start to get slightly annoyed. I’m only slightly reciprocating within the conversation and hoping he will take the hint. Another hour goes by and Jake has still not taken the hint that I don’t want to talk anymore already saying I’m going out to get food, which he then invites himself to. We get back and he’s still talking my ear off while I’m eating. At this point it’s been almost two hours and I’m sitting on the couch because I know if I go to my room he will follow me unless I say I’m going to bed, which I had no intention to. I am giving him almost no responses with a head nod every once in a while to not be completely rude. Eventually he finally leaves because my roommate grabs his attention, but this is not the first time this has happened.

Would I be wrong for telling my roommate’s friend to shut up?
```

**Verdict:** KEEP

---

## 448 · `de6c6d0c36c968e3` · 597c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for not going to SIL's wedding?

```text
My SIL is going to be getting married next year and they have chosen to have a destination wedding. She just sent us the details, and it's going to cost us $2500 each to go. We hinted to her that we might not be able to go due to costs, and she totally freaked out. For some background: husband is still a student and I am hoping to go back next fall. We both love SIL very dearly and have a close relationship with her, but we didn't even spend $5000 on our own wedding! It's going to be a big financial strain on us, and I don't know what to do.

Would I be wrong for not going to SIL's wedding?
```

**Verdict:** KEEP

---

## 449 · `df1b07b33c8babe7` · 1276c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told my housemate to move his giant pot of food out of the shared fridge or make it take up less space?

```text
For reference, we're 5 people sharing 1 fridge/freezer, and no one shares necessities like milk or eggs (much to my chagrin just for spatial economy reasons but that's a losing battle). Fridge space is at a huge premium, even despite several of us having mini-fridges to keep stuff like personal drinks or leftovers. The newest housemate likes to cook his meals for the week, which is fine as long as we've all got the space we need, and I'm certain a week's worth of meals can fit into a small space; I've done it before. I open the fridge this morning to find a 6-quart Dutch oven full of his chili taking up a huge part of the bottom shelf. I was pretty annoyed and muttered "*five people live here*" to myself. We already have enough problems squeezing everyone's food in, and this isn't the only thing of his in the fridge; this is in addition to the tupperwares containing his meals and his other assorted perishables. I want to tell him that his giant-ass pot needs to be out of the fridge and the contents need to go into tupperware to be stacked etc. because we're really, really short on space, but I'm not sure if this is me overreacting.

Would I be wrong if I told my housemate to move his giant pot of food out of the shared fridge or make it take up less space?
```

**Verdict:** KEEP

---

## 450 · `df5be37c9b6075e8` · 1326c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I point out my friend’s blackheads?

```text
Ok I know this sounds bad but hear me out. One of my closest friends has a problem with blackheads all over his face. I’ve known him for over a year and they’ve been there pretty much that whole time. It’s nothing crazy, there’s only about 5 or 6, but they are pretty big and noticeable and very obviously blackheads that can easily be extracted. One of them is right above the corner of his lip and is probably 1/8” in diameter, so it does stick out pretty significantly when you look at him. He also has a lot of confidence issues and probably some body dysmorphia. He’s a VERY attractive guy but hasn’t had a lot of positive experiences with girls. Personally, I think it’s because of his confidence, because if he was more confident I think girls would be all over him. He often vents to me about his insecurities but has never brought up the blackheads. I honestly don’t think he even realizes what they are or how easy it would be to remove them. Even though he has never specifically brought them up as an insecurity, I’m sure if they were gone he would see his beautiful clear skin in the mirror and feel a little better about himself. I have some blackhead removal tools and would easily, safely, and painlessly be able to get rid of them in around 10 minutes.

Would I be wrong if I point out my friend’s blackheads?
```

**Verdict:** KEEP

---

## 451 · `df6e5dce4d093d71` · 1212c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I were to confront my boss/longtime friend about complaining about his job?

```text
So my current boss is also a friend of mine. I’ve known him for over 15 years before I even got my job with him. I’ve been working my new job for just over a year and and he’s been the boss of the branch ever since it opened about two and a half years ago. It all started out great. Got great hours, decent pay, vacation. But the entire time I’ve worked here, I only hear my boss complaining about how he doesn’t get paid enough, how he falls behind on the mounds of work that gets dropped on him, how he is getting a second job because he claims to not make enough money here (but I know he’s just an irresponsible spender and he makes plenty.)  It’s gotten really discouraging for me because I have intentions on moving up in the company but it just makes it so hard to hear him complaining. I know his position isn’t as hard as he’s making it because I know he’s the kind of guy to try to make numerous short cuts that almost always end up backfiring on him and he’s also just a tad bit immature for his age. I’ve wanted to say something to him. But I just don’t know how to do it in a way that won’t offend him.

Would I be wrong if I were to confront my boss/longtime friend about complaining about his job?
```

**Verdict:** KEEP

---

## 452 · `dfd87e015222b4a0` · 1253c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I ditched out on rooming with friend for next school year?

```text
So I am in college and currently live with 3 friends that I met freshman year, and I enjoy living with them. Next year I was planning on living with 3 of my best friends growing up who also attend the same college. A few weeks ago one of them backed out and it wasn’t a big deal we were just gonna look for a 3 person house instead of a 4 person. Today another one of them backed out due to financial reasons, and now the remaining friend wants to look for a  house with just me and him. I love this friend and I have known him forever but he has had a long time girlfriend and they have talked about getting married once they graduate. She is also a great person but the problem is I feel like I will be alone in the house a lot of the time and be bored , and now I feel like I would rather just stay with the 3 friends I’m living with now cause I think I will have more fun. I just feel bad because I don’t know if my friend has other options for next year and I feel like I would be leaving him out to dry. Would it be wrong if I stayed in my current living situation and didn’t look for a 2 person house with childhood best friend after two previous plans fell through?

Would I be wrong if I ditched out on rooming with friend for next school year?
```

**Verdict:** KEEP

---

## 453 · `e016de2e3ac7ffa1` · 696c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for reporting a non-handicapped car parking in a handicap spot?

```text
So i work at a local grocery store. I close almost everyday of the week. Almost every time, I see the same car park in a handicap or expecting mothers spot with no handicap sticker. The person works for the store, and is not liked by many. Our store has lots of older clientel, so it makes me angry to see him take up the spot(usually the closest one to the door). Managers have been notified in the past, but nothing has happened. Full disclosure, I would like to say there are no personal feelings involved, but it would feel great to see him get a ticket or something like that. Is it not that big of a problem?

Would I be wrong for reporting a non-handicapped car parking in a handicap spot?
```

**Verdict:** KEEP

---

## 454 · `e0b7471803f2455f` · 1071c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told a friend that I already have something they gifted me?

```text
So my friends know I'm interested in vinyl records, and that I have a couple that I listen to. One of my friends wrapped up a record and gifted it to me. Unfortunately, since we had classes, I couldn't spend more time with her and I couldn't open it when she was there. She asked me instead to video myself opening it, since she wanted my reaction. Anyway, I did video myself. And when I opened it, I saw a record that I already had. What were the chances? Since I was on video, I didn't know how to react. I was just so surprised I said "wow, thanks so much!" and ended my video. I havent sent the video, and I don't want to send the video. I don't know if I should explain myself to her or not. I appreciate her gift so much; it's not like I hate it. And I don't want to conceal from her the fact that I already had it. tl:dr - I was gifted a record from a friend, but when I opened it, I realised I already had it, and I don't know if I should tell her that I already have that record.

Would I be wrong if I told a friend that I already have something they gifted me?
```

**Verdict:** KEEP

---

## 455 · `e0e41eec9d85774c` · 712c · ask=title

> **Title** (not part of the prompt): WIBTA if I quit my apprenticeship early?

```text
I work for a major company in the US as an apprentice, the duration of which changed from being scheduled for 2 years to 4 years. I've been there now for about 16 months and I just really dont like working there. There hasn't been a time in the last year where I've found myself happy with the job and while it pays well, I'm not happy at all when I go there. I'm not a major part of any of the projects there, I have no friends at the place, and I just dont find myself enjoying the work I do. I have the option to leave (without getting the degree) and I'm considering it.

Would I be a dick if I left? I feel like I'd be letting my coworkers down if I did.

Would I be wrong if I quit my apprenticeship early?
```

**Verdict:** KEEP

---

## 456 · `e18d034d951aeed8` · 2178c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I forced my adult sisters to save some money before releasing their inheritance to them?

```text
Hello All, So my parents gave me $120K (40K for each of us) when they got a lump sum of money from a house sale because are not responsible with money what so-ever. They gave me this money to use some for my house and give my sisters their 40K when they get married/buy a house. My parents gave me full control of this money as to when and how it will be given to my siblings. My sisters do not have plans to do either(marriage or buy a house)in the near future but they have  followed in my parents foot steps and have terrible spending habits. My parents were not good parents and I had to raise my sisters for the most part (I'm the oldest 28, 25, 24), I packed their lunches, dropped them off at school, went to parent-teacher meetings among other things. One of them (25 yr old) makes about 70K /year (Canadian), has no property except a car and lives with my parents rent-free. She has no savings, lives paycheck to paycheck and often asks to borrow money form myself or someone else because she's spent all hers. The other makes about 42k/year, no car, lives with me and pays rent, (I put aside half of her rent money to give to her when she decides to move out). She is also in the same situation as the other, lives paycheck to paycheck, spends endless amounts of money on food and amazon. Both of them have student loans under 10K each and are just making interest payments as they don't feel the need to pay this off right away for whatever reason. Here's my issue, I think that in order to teach them how to save (I know they are adults, but they were spoiled and are still my babies), I want to tell them that they should each save up at least 20K or pay off their student loans and save 10K before I release this money to them for a wedding or buying a home. I feel terrible as I should have taught them to save and spend their money wisely, but I'm not much older than them, so I didn't really have an opportunity to. Also, my sisters still come to me for advise, money issues, guidance as a parent role instead of my parents, if this has any relevance.

Would I be wrong if I forced my adult sisters to save some money before releasing their inheritance to them?
```

**Verdict:** KEEP

---

## 457 · `e1926568d90e2065` · 985c · ask=body+title

> **Title** (not part of the prompt): WIBTA for wanting my brother to replace what was destroyed under his supervision?

```text
Okay, so a couple of days ago I had to go to work and I left my pencil pouch filled with my art supplies on my bed behind a closed door. Note that my door doesn’t fully close and it used to have a latch on it but the latch broke, so there was nothing to stop my door from being opened. 

  My older brother was in charge of watching my family’s 3 large dogs (Two English Mastiffs and a Great Dane). I come home at 8 o clock and find that my leather pencil pouch, pencils, gel pen, and all my mechanical lead has been eaten and ripped apart. 

  My parents say that I should have put my supplies away and not left them on my bed but I think that my brother should have watched them better. 

  I want him to replace my supplies because he should have kept a better eye on them (although he did say that he kept them out of my room several times, I doubt that he did.)  
Am I the asshole here?

Would I be wrong for wanting my brother to replace what was destroyed under his supervision?
```

**Verdict:** KEEP

---

## 458 · `e230fcbe7c497233` · 1087c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I Disobey In-Laws?

```text
Long story short, I store my summer car at my In-Laws during the winter, in their backyard, and appreciate them allowing me to do so. Now this is the third year of winter storage. Year one, no issues getting the car out when I felt it was right. Year two (Last year), it became a heated argument because they felt I would destroy the grass (soft ground), even though year one car removal went without a hitch. Sure enough, father inlaw got the car out one day for me after the arguments. Now were on year three of winter storage, and I am again having issues getting the car out when I would like and am being told I need to wait until "They" feel its right. My possession (Car), but their property. I am on the verge of just going there and driving it out regardless if they like it or not, but at the same time, not hot headed enough to make a rash decision. Will note: my relationship with the In-Laws is meh. Not great, not terrible. Not much talking or visits outside of holiday. I wouldn't be heart broken if they were mad at me for a while.

Would I be wrong if I Disobey In-Laws?
```

**Verdict:** KEEP

---

## 459 · `e2358e6e0f7ad96f` · 757c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I lied about vacation to avoid a family friend from tagging along?

```text
I have a family friend who is pretty much family to me. for the past few years, this person has tagged along on the vacations we take. they pay their way, so it’s not about that, but this person is just not easy to vacation with. we don’t have the same interests. this person just wants to go to the bar and I really don’t want to spend my whole vacation doing that. they also have trouble walking long distances, which makes it hard to explore new cities like i’d like to. I like this person, but we are incompatible in terms of vacation. it should also be noted that this person takes several vacations (at least 5) a year on their own, whereas I take one, MAYBE two.

Would I be wrong if I lied about vacation to avoid a family friend from tagging along?
```

**Verdict:** KEEP

---

## 460 · `e24ea78291debb5a` · 941c · ask=title

> **Title** (not part of the prompt): WIBTA for hiding or throwing away roommate's disgusting sweatshirt?

```text
I know the title is weird so I'll explain the situation: 

My roommate has gross tendencies (generally unclean), and he has been wearing the same sweatshirt for about a month now. when I say this, i literally mean that he has been wearing this sweatshirt almost every day for an entire month. and i mean like wears it out in public, just around the house, AND when sleeping. and now I know he hasnt washed it because he was just talking the other day about how laundry hasn't been done in like 2 months. 

he's starting to legitimately smell because of this sweatshirt. there's no sign of any form of motivation to wash it any time soon. i'm getting really disgusted. so would i be an asshole for hiding or throwing it away next time he's in the shower and leaves it in the room. i mean the poor thing is stained as all fuck and smells like something fermented.

Would I be wrong for hiding or throwing away roommate's disgusting sweatshirt?
```

**Verdict:** KEEP

---

## 461 · `e4464e10be955b57` · 556c · ask=body+title

> **Title** (not part of the prompt): WIBTA for going to a food drive for people affected by the shutdown when I don’t need to?

```text
I am a member who is not receiving a paycheck due to the shutdown, however I have been fortunate enough to be able to put enough money away to cover all of my expenses for about 2-3 more months. There was a food drive available for everyone at work to attend to, and I decided not to go because I don’t feel like it was necessary for me to. 

I guess I’m just curious on the ethics/morality of my decision if I had decided to go.  what are your thoughts?

Would I be wrong for going to a food drive for people affected by the shutdown when I don’t need to?
```

**Verdict:** KEEP

---

## 462 · `e536214f6371a66e` · 1058c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I ask my dad's girlfriend to either put a bell on her cat or not let her out at all

```text
Dad's girlfriend moved in with us a little over a year ago. She brought her cat with her. The cat doesn't use a litter box because she spends most of the day outside and comes in when it gets dark/cold. The problem I'm having is last year she was an absolute terror on the small critters around us. We live in a small suburban neighborhood, and all of our backyards are divided by chain link fences. Just about every day she would kill a bird or squirrel and it wasn't long before we just had no birds in our backyards. The neighbors aren't happy. We've been told by multiple of them that they don't appreciate having a cat running around their back yards killing the birds that they enjoy having around. This area has never had a problem with this as long as weve lived here and my dad's gf seems to not see a problem with it. My solution was that we could put a bell collar on her and that would serve as a bit of a warning to the animals around her.

Would I be wrong if I ask my dad's girlfriend to either put a bell on her cat or not let her out at all?
```

**Verdict:** KEEP

---

## 463 · `e58c2a57e661f0d1` · 563c · ask=body

> **Title** (not part of the prompt): WIBTA Headphones

```text
Would it be rude if me to ask my roommates to use headphones when in the common areas? Luckily they don’t listen to anything inappropriate, however it is kind of annoying when I am trying to sleep in or watch tv, they listen to music which isn’t too bad , but I’d rather not listen to it they also FaceTime their families, which is why I am hesitant to say something cause I don’t want it to come across as rude and I don’t want them to feel bad for talking to their families... Please let me know if I am in the wrong/ being too petty about something so small...
```

**Verdict:** KEEP

---

## 464 · `e5ab2e0588c22f5b` · 695c · ask=title

> **Title** (not part of the prompt): WIBTA if I got a henna tattoo because my friend asked me to?

```text
I’m a white female, and my friend is a Muslim female. She’s apart of a club about celebrating Islam in our school, and they’re having a day where they do henna tattoos for other students as an event to promote awareness about their club. She’s asking me to get one, which I’d love to, but I’m afraid that it’d be considered cultural appropriation. 
I go to a mainly white, liberal school, and I’m scared that people will think I got one without knowing the meaning behind hennas and I’ll be ostracized. 
I don’t want to offend anyone by my actions, but getting a henna seems fun and my friend is specifically asking me to.

Would I be wrong if I got a henna tattoo because my friend asked me to?
```

**Verdict:** KEEP

---

## 465 · `e5ac00edeb6626f0` · 1100c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked again?

```text
For context, I was best friends with this girl until last year, we'll call her Carol. We had a falling out a while back (that's a story for another day), but I've always been nice and polite to her despite everything. I lent her a 50th anniversary, limited edition copy of one of my favorite books of all time, The Outsiders. I lent Carol the book in November 2016, 2 and a half years ago. She still hasn't given it back. I asked about two months after I lent it to her. Then, I asked again last year. No luck. I asked again last week and she just said, "oh it's in my house somewhere." I texted her last night and asked when she thought she would be able to give it to me. She didn't respond. Then, I asked her if she got my text today and she said "yeah" and then walked away in a hurry. I know I can just get it from the library if I want to read it again, but I paid $20 for that book and I really love my books. It takes me a lot of trust to lend someone a book. She hasn't given me a concrete answer about where my book is in 2.5 YEARS. It's driving me nuts!

Would I be wrong if I asked again?
```

**Verdict:** KEEP

---

## 466 · `e61f58f72b58738b` · 501c · ask=closer

> **Title** (not part of the prompt): WIBTA is I pre-planned my funeral and paid for a simple direct cremation with no service?

```text
I am considering pre-planning my funeral and I want to avoid as much financial strain on my family as possible. I found out that in Ohio you can get a direct cremation with no service for about $700, so it wouldn't be too hard for me to pay for that. 

I am wondering if that would be a shitty thing to do to my family as they would not have a professional memorial service. I wouldn't tell my family about this plan until after I am dead (through a will or a letter of some sort.)

What would you do?
```

**Verdict:** KEEP

---

## 467 · `e636c19188b8f2b9` · 735c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if i cut off a potential relationship because he still hangs out with his ex

```text
I (F18) have been on a few dates with this guy and my feelings are really starting to grow, however, he (M18) just got out of a two year long relationship and even longer friendship. He swears he’s over her and everything, which is cool but partly unbelievable to me. They hang out actively and it makes me feel a bit weary. Is this just insecurity and me being dramatic, or a red flag? His exfwb is also really upset him and I are talking and it’s just been a lot of drama. I really like him but I’m almost to the point where I kind of just want to let it fade because I don’t feel comfortable with him being SO close to his ex and exfwb.

Would I be wrong if i cut off a potential relationship because he still hangs out with his ex?
```

**Verdict:** KEEP

---

## 468 · `e67b66168b40b88f` · 1163c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I bury my neighbors driver's side door in snow.

```text
I live in a townhouse and my immediate neighbors have three cars but only two driveway spots. In the winter (Nov 15 - April 1) we have street parking restriction where you cannot leave your vehicle on the street between the hours of 2am - 6am due to street ploughing. They have neglected this rule and done just that on numerous occasions. They park their car between our two driveways. When plough comes by at night, it has to steer to avoid their car leaving a massive windrow of snow in front of my driveway stretching out 1 m into the road. So now I am tasked with clearing my driveway and half the street if I want to make it out of my own driveway. The local parking enforcement is useless. Next time that happens, I want to take matters into my own hands and use all the snow I cleared from the street (not my driveway) and pile it next to their car door. Side note: one time this happened last winter, as I was shoveling the damn street, the neighbor came out. Laughed at me. Got into their car (the one that was still parked on the street) and drove off. We are not on good relations.

Would I be wrong if I bury my neighbors driver's side door in snow.?
```

**Verdict:** KEEP

---

## 469 · `e68a212db8c24615` · 456c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I turn the thermostat down when I'm out but my lodger is in

```text
Right nice and simple, I'm the owner of the house I have a lodger who pays slightly below market rate by about £25/$32/€28 a month to live here. He's not always in during the evening, I'm out for the next three evenings. It's normally around 22c/72f and I'd knock it down to about 18c/65f purely because I don't want to be wasting energy heating a house that might be empty.

Would I be wrong if I turn the thermostat down when I'm out but my lodger is in?
```

**Verdict:** KEEP

---

## 470 · `e7320643800da794` · 630c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I snitched on a drug addict?

```text
We’re in 9th grade, and I heard a rumor about that he did a ton of drugs so I asked him about it, and he spent about 10 minutes describing how he does adderall and lsd and he smokes weed. He wasn’t really discrete about, just instantly said it. Afterwards he asked me not to snitch because somebody already had once before and he had to say it was a joke. Thing is, he doesn’t have his shit together at all, he’s never got anything done and I’m pretty sure he’s failing a few classes. I want him to get help, but I’m not sure if snitching would get him help or just ruin his life.

Would I be wrong if I snitched on a drug addict?
```

**Verdict:** KEEP

---

## 471 · `e732ee9ce21dd76a` · 1072c · ask=body+title

> **Title** (not part of the prompt): WIBTA for killing one of our chickens?

```text
Unfortunately the chicken coop for our hens isnt as secure as we thought. Recently we have been noticing eggs missing for a while and then in the past week discovered one of our hens had been attacked by a possum. The poor girl lost almost all the feathers on her back and it is scabbed over and has an injured leg and is unable to walk or really move well. On top of all this we figured out today she has mites so I covered her in dust treatment to try to hopefully rid the filthy bastards. When we discovered her injured we brought her inside the garage in a small box with food and water. Surprisingly lola has a bright attitude and is eating and drinking but unfortunately her leg or fatherless back doesnt seem to be doing much better after about a week.. what do you think? Should we just let her go? We love her and want her to live but don't want her to suffer. Its hard to decide because she doesn't act like shes in pain...  Note: she is still covered in mites despite the bath so we will continue to treat her.

Would I be wrong for killing one of our chickens?
```

**Verdict:** KEEP

---

## 472 · `e8528cc3ecc1d7e2` · 1024c · ask=title

> **Title** (not part of the prompt): WIBTA if I told my friend he was being a dick?

```text
I broke my toe, like an idiot, right before a wrestling match, and now I’m out for the season. Not a big deal right? Well I was having a conversation with my friend, and then he brought up my toe. I thought, “Ok, here come the jokes about how much of an idiot I was”

He then proceeds to tell me that I’m being dramatic, and that I should’ve just taped it to another toe and not gone to the doctor. He said it’s partly my fault that I have to sit out since I went to the doctor. And while I agree that it’s my fault that I broke my toe, and think it’s unbelievable that he’s saying that I chose to go to the doctor. Of course I told my dad my toe hurt, and that it was bruised black and blue to all hell.

I guess I just feel like maybe he’s partly right, and that I am being dramatic. And I don’t want to cause drama in our friendship because we’re pretty close. I also don’t want to fuck up his relationship with his girlfriend, as he’s mostly a pretty great guy.

Would I be wrong if I told my friend he was being a dick?
```

**Verdict:** KEEP

---

## 473 · `e913ac3c8b86cabb` · 1167c · ask=title

> **Title** (not part of the prompt): WIBTA if I got one a gift but not the other?

```text
I'm a "middle child" and have 2 siblings - an older sibling who is turning 32 in a week from now and a younger sibling who is turning 22 at the beginning of March.

 I have always bought presents for both of my siblings on their birthdays (as they do for me). 

Now that my older sibling is turning 32 I don't really feel like doing the whole birthday present thing anymore but I still want to buy a present for my 22 year old sibling....I feel like I am going to look like an asshole if I don't get my older sib anything but then a week later turn around and do something for the younger one. I dont want to make my older sibling feel uncomfortable or unloved ..its nothing personal...but...like when is the cut off??

I thought about possibly just doing something small for my older sib but once again dont want it to seem like I put a bunch more thought into the younger sibs gift...

Would this make me an asshole? 

TLDR - would I be an asshole if I bought my younger 22 year old sibling a birthday present when I didnt buy my older 32 year old sibling a present when their birthdays are only a week apart

Would I be wrong if I got one a gift but not the other?
```

**Verdict:** KEEP

---

## 474 · `e9934b28f364830a` · 910c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for leaving a potential roommate grouping because I found my own housing solution?

```text
A group of people and I (4 people total), all young professionals, will all be moving to a new city and met via Craigslist to meet and move into a rented house together in a couple of months. However, we have run into a lot of trouble due to restrictions about housing (single family neighborhoods so we need a special permit to all live together) and have had some difficulty finding a suitable place that fits all of our needs - location, etc. However, I have seen more listings via Craigslist about existing houses that need just one person to fill a room, etc. and have reached out to some of these as well. It shouldn't be impossible to find a place with this group, but it will be much more difficult, whereas if I find a space for just myself it will obviously be simpler and easier (and cheaper, too).

Would I be wrong for leaving a potential roommate grouping because I found my own housing solution?
```

**Verdict:** KEEP

---

## 475 · `e9bb79ecb608d694` · 1114c · ask=body+title

> **Title** (not part of the prompt): WIBTA If I quit my job without notice.

```text
I work a fairly normal part time job, but we recently got a new boss who hates me and has been trying to fire me. 

Earlier this week, he apparently changed the schedule to have me work closing shift tonight (I check the schedules when they come out and this shift wasn’t on it). He also did did not notify me that he added the shift.

He is fully aware that I have track on week days until 7:30 and that I couldn’t come in at 4:00 even if I wanted to, it was one of the first things I told him when he became the owner.

He told me today that I had missed the shift and  that I was unreliable and should have told him I couldn’t make it. He also said that since this wasn’t the first time he should fire me, and implied that I was lucky to have a job (I have never missed a day before, except for when I had a stomach virus and couldn’t come in, which I called to tell him about).

He has also scheduled me on days I had called of on multiple times forcing me to reschedule plans with friends.

So would I be the a**hole if I quit without giving him any notice.

Would I be wrong if I quit my job without notice.?
```

**Verdict:** KEEP

---

## 476 · `e9c50052e2e29a9d` · 1190c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for making him pay for my eyeglasses?

```text
So I've been seeing this guy for about a month now. He has a 6 months old golden retriever puppy that I love very much, but since it's a puppy, it has a lot of energy and jumps a lot on me. When I'm at his place, I try as much as I can to push the puppy back and tell it "no" when it's too hyper, but it always come back to scratch (not on purpose) and lick me (totally on purpose but it's okay). Today I noticed a scratch on my eyeglasses that wasn't there yesterday (before going to his place) and it's bothering my vision when there is too much light. I have very, very bad eyesight so I can't function without my glasses. I showed the scratch to the guy and told him it might be because of his puppy. He seemed very bothered and said that he would pay for the lense. I have not asked him to do so, but I'm a poor student and really need to change the lense. The two lenses together costed me about 350$. I would feel really bad for making him pay for the lense since I can't be 100% sure that it's his dog fault (although I'm pretty sure it is), and even if it is, I may be kinda responsible for not pushing the puppy back enough.

Would I be wrong for making him pay for my eyeglasses?
```

**Verdict:** KEEP

---

## 477 · `ea0c9c1ac2fc53dd` · 885c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA for inviting other friends onto a project I originally created with someone else?

```text
Genuinely conflicted here. I have a good friend who is flaky as all fuck. We came up with a drunken idea over Skype (she’s in a different country) for a podcast. Since then, I’ve been seriously working on it and have a solid idea and plan in place. Predictably, she hasn’t responded to any of my messages and probably won’t for months. I have another friend who is super reliable and lives in my city. She has another friend who would be a great addition as well. I feel so conflicted. I have worked hard on this idea and want to actually do it. If I wait for Friend A, it’ll probably never happen, or will be super sporadic. However, I feel like a total ass for considering presenting the project to someone more reliable. Help! What do you think? Or should I give Friend A more time?

Would I be wrong for inviting other friends onto a project I originally created with someone else?
```

**Verdict:** KEEP

---

## 478 · `ea289bd52a841a2e` · 1013c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I asked my dad if I could return the guitar he got me for Christmas

```text
Let me start of by saying that I am very grateful my parents have good enough jobs to buy be presents like they do. So for Christmas last year my dad got me a guitar. I have only expressed wanting to play guitar once 2 years ago, I have never mentioned it since and I never asked for a guitar. I only ever mentioned it in the first place because he asked me if I were to learn an instrument what would I chose. When he bought me the guitar I faked excitement and thanked him for the gift. I have tried to play it 5 times and did not enjoy it a single time. It is still in perfect condition and would have close to the original resale value. Lately he has been trying to force me to play it and I have been making excuses to get out of it. It honestly feel like he bought me a chore. If we were to resell it I would be fine giving him the money or using it to buy a cheap keyboard (an instrument I actually have interest in)

Would I be wrong if I asked my dad if I could return the guitar he got me for Christmas?
```

**Verdict:** KEEP

---

## 479 · `ea489ecb4462bdad` · 686c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA If I stopped talking to my friend because her boyfriend was a dick to me?

```text
Long story short I never really liked him, he would usually poke around his nose in my business and he doesn't even know me. We never talked much because of it. I've already told her a lot of times that I don't enjoy his company and that I'm not comfortable to talk to him. So last night I was talking to her about a girl that I like but I didn't know he was the one using her account and talking to me. He later was taunting me that the girl is way outta my league and stuff. Later my friend told me that he was the one using her account but she didn't really apologize for it and it felt bad.

Would I be wrong if I stopped talking to my friend because her boyfriend was a dick to me?
```

**Verdict:** KEEP

---

## 480 · `eb41683de3001308` · 2025c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if a gave a guy a taste of his own annoyance?

```text
Since I’m majoring in double media, I’m in the media classroom (we only have one) for more than twice a week and so is this guy who is also double majoring, I guess. The way our desks are situated means there are 4 PCS on one table, two on each side. I sit in the back next to no one else which is a bonus but this ~~fuck~~ dude sits right in front of me, where I can’t see him. For some reason, he loves to rapidly bang on the table, which is the most annoying feeling in the world. Either furiously tapping, legitimately banging on the table sporadically or swinging his chair against it, he does it *all the time.* I don't know if he’s doing it on purpose but it’s so loud and so distracting and for multiple occasions, I’ve asked him to stop politely because I want to actually mind my own business. When I ask him, he doesn't continue for the whole lesson and I thank him. And I thought he wouldn’t do it again after the first time. But no, *he continues every lesson from then on.* I don’t want to sound intrusive or be a bitch but when it genuinely distracts me from actually doing work, my demands get more agitated every time I have to tell him to stop. The thing is, nobody notices and I don’t want to bother the teacher with it because it seems so menial and stupid to think about. I seem to be the only one actively getting bothered by it and it’s so annoying. I also can’t move PCs because the classes are packed so there are no free computers and I legitimately want to mind my own business without any interruptions. Just to give him a taste of his own medicine, for a whole lesson. Or at least do something about it without seeming like it’s a big deal? Just once though because I really don’t want to be seen as a dick, ~~like he is.~~   **TL;DR,** this dude keeps banging on the table distracting me and I politely tell him to stop whenever I can. The dude is still going so should I do the same just to give him a taste of how annoying it is?

Would I be wrong if a gave a guy a taste of his own annoyance?
```

**Verdict:** KEEP

---

## 481 · `eb526b7ab784ff54` · 1203c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I get a pro haircut?

```text
Ok hear me out. My hair is currently very long, and my housemate is a hairdresser. She recently was like 'Oh I'll cut everyone's hair!' and she already did my other housemate. Great job, all good. Now the problem. I've been wanting to go super short for a while, which is what I used to have. So I asked if she could do it? And also I offered to pay if she wanted since giving a short haircut to a woman can be a lot of work. Instantly I could tell I made a mistake. Her eyes got huge and she went into diplomacy mode. Oh she'd go short, but let's do medium-short because I can always go shorter, right? Oh crap, I think, she doesn't want to give me a really short haircut and then live with me if I get offended. I scared her. No amount of insisting would convince her that actually I won't care however it turns out. In the end she gives me a cut and cuts off literally an inch, and my hair went from very, very long to merely very long. It's nice, but?? My friends laughed at my new short hair when I showed them. Now my question, I still want *actually* short hair, and I don't want to put pressure on my housemate. Or am I stuck now with long hair forever?

Would I be wrong if I get a pro haircut?
```

**Verdict:** KEEP

---

## 482 · `eb8fb4719c6ccb09` · 1144c · ask=body+title

> **Title** (not part of the prompt): WIBTA For not attending close friends surprise party?

```text
I'll call my friend Dave for this story. 

 We've known each other for half of each other's lives and while I don't think I'm his closest friend, I'd say we're pretty close.  His birthday party is coming up, and his parents invited me and some of our mutual friends to join in on a surprise party for him. 

The party is on the same day as one of my first games, and we have already had 3 games cancelled dude to bad weather, I really don't want to miss the first one. The party is a long drive away as well. It's about 2 hours away from me, we live nearby but the party takes place far away. 

So I'm stuck. I want to be there for him on his 18th, but I also don't want to miss my first game and drive 4 hours total. My original plan was to go and be there for him, then leave early to make the game, but the times conflict. 

What I'm going to try an do now is go take him out somewhere nice for his birthday, that way he still has fun at his birthday party, I'm celebrating it on another day, and I get to go to my first game. Is that me being selfish and the asshole though?

Would I be wrong for not attending close friends surprise party?
```

**Verdict:** KEEP

---

## 483 · `ebf9c72a79278d9c` · 816c · ask=title

> **Title** (not part of the prompt): WIBTA for not helping a classmate?

```text
Okay so here it goes. This girl befriends people who may be useful to her in the future in general. Not only that she won’t ever help anyone, she talks highly of herself to everyone around her but yet still pretends to be nice. So it is not possible for others to address her arrogant behaviour.

I feel like she only talks to me because I would be useful to her. Having been used by such people in the past, now I’m following a social policy in which I only help people who I’m sure would return the favour. 

Last night, she sent me a message wanting my help for a project and I still haven’t replied. I don’t know if I am wrong for not helping her and to be honest I want her to learn a lesson. 

Sorry for any grammatical mistakes, English isn’t my mother language.

Would I be wrong for not helping a classmate?
```

**Verdict:** KEEP

---

## 484 · `ec1914fc60876fec` · 402c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I tell my father about a fake gift?

```text
I've been gifted fake perfume from my father's new girlfriend, totalling about £300 if it were real. The gift itself is lovely, and I'm glad she thought about us so I'm not about to mention anything bad about the gift. However, my father is under the impression that it is real and thinks it's such a wonderful gift and won't stop mentioning it.

Would I be wrong if I tell my father about a fake gift?
```

**Verdict:** KEEP

---

## 485 · `ec6523808d1004f1` · 1456c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for not attending their Christmas party?

```text
I have a large family with divorced grandparents and this means that every year for the entire week of Christmas I have 3-4 family gatherings to attend depending on the year. My stepdad has been in my life for 12 years now...him and I have a lot of differences and have hit heads a lot in the past - we can get along okay and be cordial with one another but we really don’t have any kind of relationship outside of family gatherings.. My  stepdads parents live far away from us and we only see them once a year usually around the holidays..In the 12 years I have known them I have probably only seen them 12 times but they have always been kind to me when we get together and they even send me birthday cards every year (which is pretty thoughtful and more than my biological family does)....I generally enjoy being around them... this year from the 21st to the 26th I have 3 different Christmas get togethers to attend with my biological family and then my step dads parents get together is on the 29th. In all honesty I just don’t want to go...it is nothing personal towards them but after a full week of family get together I wouldn’t mind a day or 2 for myself while I have the time off of work and really don’t feel like taking the 4 hour drive....part of me thinks “well my stepdad and I don’t even really talk anyways” but then another part of me feels that it is disrespectful to not go...

Would I be wrong for not attending their Christmas party?
```

**Verdict:** KEEP

---

## 486 · `eddea6077cccb9dc` · 486c · ask=title

> **Title** (not part of the prompt): WIBTA If I paid for a Switch with 125 bills?

```text
I recently rediscovered a sum of money that I was saving up a few years ago, and wish to spend it on a new Switch. If I go to gamestop and pay with a ludicrous amount of bills, would I be an asshole? I would really prefer if I didnt have to deposit it into a bank but that's really the only other option.
I think they will take the money anyways, but I dont really know if doing something like this would really cause any issues.

Would I be wrong if I paid for a Switch with 125 bills?
```

**Verdict:** KEEP

---

## 487 · `ee51a2ac4565c90a` · 517c · ask=body+title

> **Title** (not part of the prompt): WIBTA if i got my cats declawed?

```text
title says it. all my childhood i’ve had cats and my parents chose to declaw them as kittens. now that i’m old enough to have my own cats, should i get them declawed? everyone acts shocked when i tell them this saying how cruel it is, but my cats have always seemed fine. they’re indoor cats and don’t need to defend themselves against any prey. what are your thoughts??

also i love love love my pets so please don’t be mean. i really don’t mean them any harm. thanks. :)

Would I be wrong if i got my cats declawed?
```

**Verdict:** KEEP

---

## 488 · `eed8c35c5e9eb288` · 574c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I take an all expence paid business trip?

```text
My current employer is having a huge ”owners” meeting in an area I have never been to before. The owner of my franchise location invited me and is paying for EVERYTHING! This includes the flight, airfare(never flown before), food, and transportation. I am really looking forward to this trip, however, I'm also considering putting in my resignation a couple of days after I get back. I wouldn't say this to them but business isn't doing great and I just got a pretty great offer to move into a whole new career.

Would I be wrong if I take an all expence paid business trip?
```

**Verdict:** KEEP

---

## 489 · `efcfeaa16a596045` · 1097c · ask=title

> **Title** (not part of the prompt): WIBTA if i asked them to make their kids stop screaming?

```text
Short term post, idk mow much attention this will get in the time frame needed, but here goes:
 I'm staying at a hotel, we got in at like 3 am from a long road trip, we spent over $115 just to get half a night of sleep at a clean, peaceful, comfy, quality hotel with a decent breakfast, and they put us in the room next to the pool. So far so good. 

BUT. From about 9am onwards (and today is Saturday I might add), this group of like 5 or 6 kids have been running around the pool literally SCREAMING. And the adults were using their yelling voices too. It's happy yelling but like damn. At the top of the lungs is not necessary. This has been going on for at least an hour now. We can't be the only people here who were woken up by their noise. And I've never seen a group of kids just scream nonstop like they are, and the adults join in, at a nice hotel like this.

I'm afraid to confront them because if the parents are screaming too, they will probably just get angry and tell me to stop ruining their vacation or whatever.

Would I be wrong if i asked them to make their kids stop screaming?
```

**Verdict:** KEEP

---

## 490 · `f03174da6fa7dfb9` · 790c · ask=body+title

> **Title** (not part of the prompt): WIBTA for telling the bride she should take back her request? She wants the bridesmaids to buy matching shoes.

```text
Just attended her bachelorette party last weekend and realized how much the other girls are struggling financially. Should I say something to the bride?? 

Most of the bridesmaids couldn't pay up front for the $150 dress and had to set up layaway plans. Half haven't booked accommodation for her summer destination wedding (I found a cabin for $700 back in January). At the bachelorette party, one girl was upset by $6 club sodas, and splitting the check after dinner was an entire ordeal.

Now she's asking us to buy matching heels. Is it my place to say something? Is it worth speaking up? The shoes will be ~$50, which is miniscule compared to the other expenses.

Would I be wrong for telling the bride she should take back her request? She wants the bridesmaids to buy matching shoes.?
```

**Verdict:** KEEP

---

## 491 · `f093f51fc15154ec` · 493c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I was to snitch on the girl whose house I clean?

```text
I clean houses for a living. Today I was cleaning my friends house, and when I was cleaning her daughters room, I found some stuff. I found marijuana and a few water bottles with alcohol in them. I am pretty close with this friend as we go to church together, and our families have been friends for years. I feel like I owe it to my friend but I don’t want to regret it later. I just want to make sure her daughter is safe.

Would I be wrong if I was to snitch on the girl whose house I clean?
```

**Verdict:** KEEP

---

## 492 · `f0b5a148792007d3` · 898c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for getting a septum piercing even though my family hates them?

```text
I’m 18, at home for the next year getting my massage license before heading to college. I’m very very close to my family and value their opinions—except for one thing. I have multiple piercings and several tattoos. My family have always been pretty lax and open about hair dyeing and moderate (nose/ear) piercings. However, they despise any others. But I love the look of septum piercings. My friend got one recently and I loved it. My family (and a few friends) hates it and think it looks awful. I have been having some problems with self-confidence since it’s freezing here and I only feel like wearing bum clothes. In my opinion, a few piercings kind of make it look more like bumming around is intentional rather than I just gave up on life. Also my parents are on a trip for a few weeks and would have no idea.

Would I be wrong for getting a septum piercing even though my family hates them?
```

**Verdict:** KEEP

---

## 493 · `f1326a10d350a6b2` · 794c · ask=title

> **Title** (not part of the prompt): WIBTAH if I left during a 9 hour shift in work in which there is no coverage?

```text
So (21F) i work in this bakery that’s for some reason so severally understaffed today. Saturdays happen to be  our busiest days we usually have two people working so that someone isn’t stuck there from open to close for 9 hours.

Well one of the girls called out to go to a coworkers husbands funeral from her old job and couldn’t find coverage. Which means I have to stay there from 10am-7:30pm with no breaks. Period. This is especially frustrating because when the kitchen staff all leave at 2:30 I have zero means of getting food for the next. 5 hours.

So here’s my questions; WIBTAH if I left around 3:30 and told them to find someone else  to work the rest of the shift? Because this is ridiculous.

Would I be wrong if I left during a 9 hour shift in work in which there is no coverage?
```

**Verdict:** KEEP

---

## 494 · `f15ac26be7b131ed` · 802c · ask=body+title

> **Title** (not part of the prompt): WIBTA If I play explicit music for kids?

```text
I gave birth to my first child a day after my birthday. 

Instead of having a traditional baby one year old party for my daughter (who won't care cause she is a toddler and will still get cake) I am having a joint birthday party for her and me.

I have invited all my famy and friends and made it clear that it's for both of us and there will be booze and food and cake.

I am making a playlist of songs and have just realized a lot of my music has lots of cursing. 

I know my in-laws and thier kids will be here should I steer clear of good music because of language?

I honestly don't think little kids understand cursing I mean they don't even talk themselves so...

And it's my house and my party and I don't sensor my music around my daughter.

Would I be wrong if I play explicit music for kids?
```

**Verdict:** KEEP

---

## 495 · `f1a255f3957efc24` · 900c · ask=body+title

> **Title** (not part of the prompt): WIBTA for wanting to distance myself from a new friend (potentially toxic)

```text
So I made a new friend about a month ago and she is super nice and friendly. However lately all she’s into is going out with guys and hooking up with them. She also always offers to buy me stuff and I always decline because I have my own money etc and don’t like owing people stuff. The other week she bought me a small $10 item I was looking at without telling me (I feel like she’ll expect me to owe her in the future) We also went out shopping the other week and she stole a $5 soda which i was completely against and she just shrugged her shoulders. Now I feel like I owe it to her to stay her friend because; 
1. I feel like I’m in too deep
2. She hasn’t got many other friends here 
3. I’ve seen her break friendships off and they were super messy. 
Would I be an asshole if I slowly faded away from her???

Would I be wrong for wanting to distance myself from a new friend (potentially toxic)?
```

**Verdict:** KEEP

---

## 496 · `f314965017e4fc63` · 635c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for telling a teacher about a student who cheated?

```text
On my most recent test in US History, I got an 85/100. My teacher curves the score of the test out of the highest grade achieved in the class. The student who received the highest grade is a 92, so I really got an 85/92. I know that the second highest grade was also and 88. I also know that the student who received the 92 cheated on the test, so really my grade would likely be an 85/88 and not an 85/92. I think it is unfair that this student cheated and everyone else who got a lower grade will do more damage to their overall grade because this student cheated.

Would I be wrong for telling a teacher about a student who cheated?
```

**Verdict:** KEEP

---

## 497 · `f325928b190a5a0c` · 1334c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for planning on making my roommate pay the entire electric bill?

```text
So when fall semester ended I went back home to work and get money and my roommate stayed at our apt by himself. Now all the previous times I lived at the apt myself I never charged my roommates (previous and current) for electric when they weren’t actively living there (the electric is in my name, and I lived there over the summer when my roommates would be at home). But when I left for winter break, my roommate stayed at our apt. He has a habit of not being very energy efficient (he will leave our front door wide open, turn the heat on full blast when it only needs to be on low, leave every single light on that he can find, play video games until 5am etc). I’m not the richest person and I’m half expecting a $180 bill for electric that someone else used. When I get the bill I’m planning on just handing it to him and having him pay it all since I haven’t lived there the past month. I took out all my electronics, cleared my stuff out of the fridge when I left, the only thing I did was put my heat on low in my room so my pipes wouldn’t burst. I would be fine sharing the bill if we both left for the winter, but I don’t think I can afford to pay for his expensive habits (if I’m not there to be his “mom” he won’t take care of the place)

Would I be wrong for planning on making my roommate pay the entire electric bill?
```

**Verdict:** KEEP

---

## 498 · `f36d3c86a78b2d2d` · 464c · ask=body+title

> **Title** (not part of the prompt): WIBTA If I tell my friends to stop taking about their periods?

```text
There are two friends I like talking to and they're really good friends, but happy the time so they talk about is their period. They don't make me uncomfortable or anything I get that it's a natural body function, but it's just kinda annoying and gross, and they have other times where they could talk about it not around me. I just don't know if it'll seem like I'm just being a dumb jerk

Would I be wrong if I tell my friends to stop taking about their periods?
```

**Verdict:** KEEP

---

## 499 · `f3a38838c7fe6658` · 1570c · ask=title

> **Title** (not part of the prompt): WIBTA for posing as a customer to put a complaint in about my coworker?

```text
I’ve been working at this place for years and in the past 6 months we hired a new girl, I’ll call Marie. Marie has been a problem from day 1. She’s rude to customers and staff, flat out refuses to serve customers when asked, is never working where she should be, wanders aimlessly to avoid customers and has a terrible attitude. Unfortunately she’s the only one willing to work as late as the store closes (unless we hire someone new). 

I spoke to my direct manager about this. She was infuriated as she’d received a verbal customer complaint and we’d had a contractor complain about her too. My manager was ready to fire her but needed someone to replace her shifts. That manager has now gone on indefinite leave. 

The store manager, who I’ve also relayed this to has already spoken to Marie about her constantly coming into work late (I was unaware of this) but hasn’t been taking anyone’s complaints about her seriously. 

The one thing the company does take seriously is customer complaints online. They have to go though the management chain before they reach our store so higher up managers are also aware of the situation. I’m at a loss for what else to do but I cannot work with Maire anymore, knowing she’s been spoken to by management about her attitude and work ethic but won’t change. 

However I’m quite conflicted about possibly ruining a girls career and impersonating a customer to do so but I don’t really know what else can be done to get managements full attention.

Would I be wrong for posing as a customer to put a complaint in about my coworker?
```

**Verdict:** KEEP

---

## 500 · `f3be5e934ef954f4` · 1597c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I confronted my sister about not paying me back

```text
Ok so pretty simple going off the title but the backstory to this is:  Last year my sister went to South Africa to visit some family and attend my second cousins wedding. She had saved up just enough money where she should be able to get through the trip, but it would be a bit tight. A week or so before she flew out I sold my car and had a decent amount of spare cash, so before she left I told her that if she thought she was getting low or going from run out of cash to just ask me and I'll send her some. So about 2 weeks before she flies back she messages me and says she's running low on money can I please send her some. So I do, and transferred her $750 and with a message saying to pay me back when she could. All sweet. However, since coming back my sister has been in and out of work, contemplating/going to uni and hasn't had a particularly steady cashflow. Still all good, I don't especially need the cash so it's not a problem. But since coming back and working occasionally, she's gone out multiple times a week with her new bf, and it seems like every other month she's going to a concert or a music festival. I'm not sure if she's paid for these in advance or if someone else is shouting her tickets or what the deal is with this. Now like I said, I'm not strapped for cash. I've got enough money coming in that I can pay my bills and have a hundred or so a week spending money. But there's a voice in the back of my head that's recently popped up saying she should be making more of an effort to pay me back.

Would I be wrong if I confronted my sister about not paying me back?
```

**Verdict:** KEEP

---

## 501 · `f41bfd8a11b36875` · 1017c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I replace the computer I recieved as a gift a year ago?

```text
Okay, so my mom got me a cheap computer for Christmas last year for school and while I'm super grateful, she's not the greatest with technology. It had barely enough storage to run windows so I couldn't save anything to it. I've since sunk about $150 (probably half of what this thing was originally worth) into it to try and speed it up and expand storage, which has helped some but it has now developed some other problems.
The new problems include randomly restarting  (like a lot) and it doesn't like to connect to wifi. So I've been thinking about buying something else to replace it but is it too early (since this is just over a year old) to get something new? Should I just tough it out for a while longer (I have over a year left before I graduate if that makes a difference) or do you think she would understand?

Also if it makes a difference, this is not my main computer, it's just a smaller one that's easier to take to class.

Would I be wrong if I replace the computer I recieved as a gift a year ago?
```

**Verdict:** KEEP

---

## 502 · `f4776de7b755c4ab` · 1042c · ask=title

> **Title** (not part of the prompt): WIBTA If I would have my secodary school bullies beaten up by a tavern tug?

```text
First things first, I am a pretty shy guy, 18/M , and lack of social skills(from my understanding).

My classmates bullying me from the second semester of the first year. This includes and specified at gay joking ( touch my junk, come in the shower when I am in, makes me the class's no 1. target of hatred and distrust, etc .) . I am tried to be friendly with them, but after some time, they found me and everything started again.

About the man. I met him at the tavern nearby. We spoke a lot, then he asked me about my school. I told him my problem, then we talked about it and offered me that he will threaten them to stop bullying me, and he was very serious about it.

He is the hit-before-ask type, and he respects and pities  me in his way.

I have to be with them for 5 years and I have only left five months left with this bad company and their behaviour finally changed a little, but not that much. We tried to negotiate, but that didn't help.

Would I be wrong if I would have my secodary school bullies beaten up by a tavern tug?
```

**Verdict:** KEEP

---

## 503 · `f497f30d9b374407` · 812c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for getting a tattoo?

```text
A few years ago, I borrowed money from my sister. I have a better job now and just got a Christmas bonus that will allow me to pay her back in full. I asked her if I could pay her back, and she insisted that she didn’t want it, that we’re family and that I should keep and save my money. She’s much better off financially than I am and never, ever asks me for money. I pay for things when we go out together about equally. Here’s where the question comes in. My entire family is pretty against tattoos, parents especially. My sister has one we got together, but ultimately, she always tells me to save my money instead of getting them and scolds me for having so many as is. I’d save the rest of it. I only get tattoos when I have extra money for them, and this qualifies.

Would I be wrong for getting a tattoo?
```

**Verdict:** KEEP

---

## 504 · `f51077789193874c` · 550c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I put in my 2 weeks notice while on vacation?

```text
I've worked for my current company for about three years and have never had a vacation longer than a simple 3 day weekend. My department is so small and has so few people that it's hard to schedule extended periods of time off, so I currently have about 2 and a half weeks of vacation time saved up. Would it be a total asshole move to get all my vacation time approved, then put in my 2 weeks notice while on said vacation, functionally quitting and taking 2 weeks of paid time off?

Would I be wrong if I put in my 2 weeks notice while on vacation?
```

**Verdict:** KEEP

---

## 505 · `f54c3fde12c2b135` · 726c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if i stop loaning money to my dad?

```text
For context, I'm still in school and do have a job, and live with my dad. My dad is not well off financially, but he does spend a lot of money on himself. I save money a lot, and usually have a few hundred in my bank account (which is a lot for my age). My dad often asks to borrow money from me, which I let happen. However, most of the time getting money back from him on a hassle. Recently he's been asking for amounts a little above 1 hundred dollars, and I still lend him the money. The real reason I plan to stop lending him money is that he's been really irate and unforgiving lately towards me ( not even towards my brother). (Also sorry for format, I'm on mobile).

Would I be wrong if i stop loaning money to my dad?
```

**Verdict:** KEEP

---

## 506 · `f56fe482909fce50` · 1793c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I took a stray kitten to

```text
I found a kitten in front of my condo at 9pm last Thursday.  I stood outside for 20 minutes and let her climb trees hoping I’d find somebody searching for her but nobody else was outside.  

She followed me into my house and I got her food and a litter box. I just didn’t feel comfortable leaving her out to be eaten by the foxes.  She’s about 2 or 2 1/2 months old, so she is kinda prime adoption age. 

I put up flyers around my neighborhood, my friend who lives in the same neighborhood put her up on the next door app.  I never found the owner.

Problem is - I’m allergic.  She can’t stay here.  I’ve asked everyone I know and nobody wants a kitten.  My neighbor got my hopes up and said his ex and his friend wanted kittens but both of those fell through. 

I keep her in my bathroom all night while I sleep and every time I am not at home because my home isn’t kitten proof and I have a very expensive couch that she’s already ripped.

I feel bad making her stay in the bathroom but I really didn’t ask for this. I really tried hard to find her a home and haven’t had any luck and she deserves a home where she is wanted and where she can run around and play and not get in trouble for being a kitten. 

Would I be an asshole if I took her to the humane society?  I don’t like the idea because they kill animals if they can’t be adopted but she is a kitten and I think she could be adopted, I just feel really bad because I don’t like supporting shelters that kill animals, but this adorable kitten deserves a better home than mine and the longer she is here the more I feel like she must feel like a kidnap victim :( 

TL;DR: I found a kitten, can’t find the owner, I’m allergic - would I be an asshole if I took her to the humane society?

Would I be wrong if I took a stray kitten to?
```

**Verdict:** KEEP

---

## 507 · `f5c323e9a8baa17c` · 2077c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I backed out of volunteering?

```text
So there’s an event that’s being put on by an organization I work with. Last week on Friday of the members of said organization asked if I was going, and I made a comment about not really being able to afford it. She then suggested I volunteer, which would grant me entry to the event for free. I know at least one of the volunteers and it’s for a good cause, so I said I would be interested, but needed more information (like details on times, roles available, etc). She sent out an email that same day to the volunteer organizer and cc’d me on it saying I was interested. I did not receive anything further and assumed they were full up. Coincidentally a friend is also having her birthday party on the same day, so I figured it all worked out for the best since now I could attend.

The event is tomorrow. Today, towards the end of my shift, the same woman I’d originally talked to asked if I would be there. I said I never got a response. As she was leaving out the door she said she would email him- I told her not to bother since it was so last minute, but she was literally walking away from me and I don’t think she heard me. I was about to go home when I got an email for the event organizer. He apologized for the delay, and then said they still needed volunteers so he’d gone ahead and signed me up. He then said we could chat further about what role I would like to volunteer in.

Again, the event is literally tomorrow. I get that they need help, and I was willing to volunteer in the first place. But now I have other plans, and I don’t really want to cancel them. I was also a little taken aback that he just assumed I would still be interested and available, and just signed me up without asking. It’s a great event and I don’t want to be rude or leave the in a lurch, but I don’t feel like I was treated very respectfully. I’m honestly considering replying and saying that I won’t be able to go, and to please remove me from the list. Does choosing my friend’s birthday over volunteer work make me an asshole?

Would I be wrong if I backed out of volunteering?
```

**Verdict:** KEEP

---

## 508 · `f5fc91afdf950756` · 815c · ask=body+title

> **Title** (not part of the prompt): WIBTA if i date my best friend who dated my friend?

```text
me and this chick have been friends for about 3 years in college, when i meet somebody and he becomes a close guy friend. naturally, we all hang out and they end up in a relationship. they break up after two rocky years, the guy friend moves halfway across the country and we no longer talk on a regular basis. me and the chick are still here as friends. fast forward 3 years later, me and my best friends relationship (which is at nearly 8 years now) starts to heat up. she doesn’t wanna be known as a homie hopper understandably but it’s known that we like each other. we have a hell of a lot in common, the chemistry is crazy. i feel as if this is an opportunity to be w somebody special. would i be an asshole to try and pursue this a lil further?

Would I be wrong if i date my best friend who dated my friend?
```

**Verdict:** KEEP

---

## 509 · `f62190d329c099aa` · 1274c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I quit my temp job filling in for a woman on maternity leave when the the place is already understaffed?

```text
Context: 

I just graduated with a Bachelors in December. I had been applying to jobs since mid October to no avail and was getting very anxious as I have bills to pay. The only response I had gotten was a temp job at a very large company to cover a woman while on maternity leave. It paid well enough for a kid just out of college but is not something I'd exactly like to do forever. I took it because I panicked and was happy that I'd be starting employment two days after I graduated. 
I can handle the tasks but lately they have been mounting. The place and its nearby branches seem very shortstaffed. Recently some very toxic emails have been sent en masse detailing dissatisfaction with a lot of peoples work ethic. 
Recently people have been asking me to do certain tasks  and projects without any training, and asking for help is next to fruitless as everyone is so busy. The people I work with are very kind but anyone above is a nightmare with few exceptions. I am 100% regretting taking this job. I'm applying to new jobs and looking anywhere I can for an out, but I feel like a scumbag for even thinking about leaving. Am I the asshole?

Would I be wrong if I quit my temp job filling in for a woman on maternity leave when the the place is already understaffed?
```

**Verdict:** KEEP

---

## 510 · `f6f9afa4fffa7f5f` · 1423c · ask=body+title

> **Title** (not part of the prompt): WIBTA if I got my piggy packed on my MOH’s bridesmaids gifts for mine since we share some ‘maids?

```text
So confusing title sorry, but I am getting married in October, I’ve been engaged for 2 years now and have been planning for my bridesmaids robes for getting ready because cute photos and blah blah blah. 

My best friend and co MoH (she shares the title with my sister because my sister is 2,000 miles away so they are splitting the responsibilities) is getting married in June, she got engaged last December. We are both in each other’s wedding parties and we share 2 bridesmaids. She informed me she is also doing the robe thing because that’s what everyone does. It’s cute and something they can reuse. 

However I’ve talked to one of our joint bridesmaids and she said she has no need for two robes. So I am thinking about asking the other bride if I can just get the same robes for her and my sister so the other two can just reuse those. Obviously I’ll get them other bridesmaids gift, champagne glasses with their names or something otherwise equal cute and photographable (just trying to be transparent). I’m having them pick their own dresses in the color that matches the scheme so they can rewear the dress since they are paying for them (something I cleared with them when I asked them and they are all so fine with) so I want to get them something! 

Would I be an asshole or even tacky if I did this?

Would I be wrong if I got my piggy packed on my MOH’s bridesmaids gifts for mine since we share some ‘maids?
```

**Verdict:** KEEP

---

## 511 · `f8112365f23e0e1b` · 682c · ask=title

> **Title** (not part of the prompt): WIBTA for separating a kitten from his sisters?

```text
Earlier this year, a feral cat gave birth in my parents' shed. We adopted the three kittens once they were old enough to not need their mother's milk, and I helped take care of them. I moved out of my parent's house recently, and I really miss the kittens. But my landlord says I can only have one pet in my appartment. The solution I'm leaning towards is to take one kitten and leave the other two with my parents. But I'm worried that I'm being selfish. Of course, I would take good care of him and give him plenty of love and attention. But he would never get to see his sisters again, all because I was feeling lonely.

Would I be wrong for separating a kitten from his sisters?
```

**Verdict:** KEEP

---

## 512 · `fa70b388f3146dc6` · 437c · ask=closer

> **Title** (not part of the prompt): WIBTA: Shipping shit

```text
I run a business and sell sneakers. As of late I’ve been putting the weight of the box and it’s dimensions a lot smaller to save on shipping cost and I just print out the label and drop it off. I’ve been doing this for months now and nothing has happened.  I also don’t purchase the labels directly through the shipping place because they are more expensive so to save more money I use sites such as pirateship or shippo

Help me decide.
```

**Verdict:** KEEP

---

## 513 · `fa93888598ac6232` · 587c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for simply not rocking up to work?

```text
I usually work night-shifts, 10pm-6am's. I am writing this there, and am on my 2nd night in a row out of 3 in a row. Right now I've got Tonsillitis quite bad and woke up after last nights night shift coughing up flem with specks of blood in it. After asking my boss for it off, he told me no, nobody can cover it and that I have to work it. So I decided I'd tough it out but really feel atrocious, same deal goes for tomorrow. I've told him I feel atrocious, I'm only a casual, and I think it's unfair I should work feeling like this.

Would I be wrong for simply not rocking up to work?
```

**Verdict:** KEEP

---

## 514 · `fac565175f975435` · 624c · ask=body+title · scrubbed

> **Title** (not part of the prompt): WIBTA if I told my brothers to shut up about being a man?

```text
Ok I always get shit for doing dumb things or playing minecraft and roblox because it is not “manly” and I hate it. And it is literally for the dumbest things like if I stub my toe and show any sign of pain because i stubbed my toe they would say “stop being a sissy, man up!” And I really hate it because all I want to do is have fun and they would Just tell me to be a man. I hate it when people say that being a man shouldn’t be stressful I should be able to have all the fun i want to have! What puts the difference between men and women’s behaviour?

Would I be wrong if I told my brothers to shut up about being a man?
```

**Verdict:** KEEP

---

## 515 · `faf4e2f2e8fbf122` · 1482c · ask=body+title

> **Title** (not part of the prompt): WIBTA for refusing the clean the kitchen?

```text
I live with my parents and the only job I'm required to complete is the kitchen. Keep in mind, it isn't the *only* job I do, it's just the only one I'm expected to do. I help around and do other jobs on top of cleaning the kitchen.

Sometimes I'm not at home (away for a couple of days or more) and when I come back, the kitchen has not been touched after it's been used. I can't even describe the state I've seen it in, it can be that bad. I spoke to my family about it and asked them to please clean it if I'm not at home or even just to get my siblings (ages 7 and 10) to do it. 

My dad works full time on weekdays and sometimes has to make calls after work and my mum works casually (sometimes working 2 shifts on 1 day) and cooks so I can understand that sometimes they're just too busy or want to rest. However as I stated earlier, I have 2 other siblings who could do it *plus* a dishwasher which most of the time doesn't get filled when I'm away.

Every time I come back to a messy kitchen I feel like it's always expected of me to tidy it up despite the times I've expressed someone else to do it while I'm away.

I'm not sure if I'm being ungrateful in this situation or if I'm justified. I'll leave it up for you guys to decide.

TL,DR: main job is to clean kitchen but when I'm away no one does it and I'm the one that has to clean it when I come back. Asked family to clean it when I'm gone but it never gets done.

Would I be wrong for refusing the clean the kitchen?
```

**Verdict:** KEEP

---

## 516 · `fb854de2be3d1785` · 720c · ask=body+title

> **Title** (not part of the prompt): WIBTA If I reported my coworker for getting high in the middle of work?

```text
Basically the title says it all. I work at a family owned pizza place and on Saturdays there’s a girl (let’s call her E) who also comes in to work. When it gets closer to closing time our boss sends her back to help us clean up (I work as a cook and she answers phones)  it all she actually does is dry dishes for about ten minutes (which is actually great) but right after she just goes out back and gets high. She’ll come back in stumbling around and laughing her ass of, doing absolutely nothing for the last hour to hour and a half of work. My boss is unaware because he stays up front but would I be the ass-hole for reporting her?

Would I be wrong if I reported my coworker for getting high in the middle of work?
```

**Verdict:** KEEP

---

## 517 · `fc857b97402ffdc5` · 645c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I don't hang out with my friend on her birthday?

```text
Today is her birthday, and she wants to do something for it. I made plans with a different friend to eat dinner today about 3 days ago. Birthday girl reminded me 2 days ago that April 3 is her special cool day, and asked what we were gonna do for it. I don't want to cancel on dinner friend, because I don't get to see her often. Birthday girl is my roommate (yes I know I don't know my roommates birthday by heart, I've always been bad at remembering them), so I see her all the time and I think maybe we can officially celebrate for her tomorrow or some other future night.

Would I be wrong if I don't hang out with my friend on her birthday?
```

**Verdict:** KEEP

---

## 518 · `fce704d74f0a1cbc` · 1251c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA for calling the cops on my neighbors kids?

```text
I live in a pretty small, but expensive for the area, apartment complex. The buildings are about 3 levels but the hallways and stairs are open. You can usually hear people talking out in the hall, but it’s usually not exceedingly loud. The problem is one family has a couple of kids, and since the apartments aren’t very big they send them out to the hallway to play. I’m not exaggerating when I say there’s no place I can go in my apartment and not hear screaming. It’s so bad that I now have to wear headphones when watching tv or playing games because I can’t hear anything. What pisses me off just as much is that they treat the hallway like a toy room. When I leave my apartment I’m stepping over a mine field of toys, shoes, and backpacks. Management doesn’t seem interested in doing anything about it, as when I call them about it they just say they’ll talk to them “tomorrow”. I can’t just talk to the parents because they don’t speak English, but it’s obvious that management is just letting it happen because they let their kids do this every single day. I’m now considering calls the cops. I hate to do this because I think it’s a waste of resources, but I can’t take it anymore.

Would I be wrong for calling the cops on my neighbors kids?
```

**Verdict:** KEEP

---

## 519 · `febc59f428d1e206` · 1490c · ask=closer · scrubbed

> **Title** (not part of the prompt): WIBTA My neighbour plays Bass

```text
Me: Young Professional just moved into an older style condo building. My neighbour: Middle age dude who play bass, music,movies, etc loudly The Building: Old over engineered walls I've recently move into an older style condo and my neighbour is loud. By loud I mean one can feel the vibrations of his music throughout my apartment and probably the ones near by. One of the neighbours on the same floor moved within a few months of living near this guy. So far I haven't really done much about the issue directly with him yet due to the fact that I moved in months ago but I haven't really lived extensively in the condo due to travel and work. Now I am home a tad more but mostly just at night and weekend for another week untill I will be doing a lot of work from home. I have not talked to him directly yet but I have banged on the wall when he is blasting bass,music, and/or music when I come home and try to sleep. He turns it down. The Building has over 28cm concrete walls I feel I shouldn't hear this guy. I don't hear any of my other neighbours. I think I am living near a musician who does not understand how bass travels. I play accordion and classical guitar. I rent a practice space or use my isolation booth He sent a note recently asking which hours of the day can he play his music. I'll attach it and my terse response to it. 1)Contact neighbours on the floor to determine if they can hear and feel this guy as loudly as I do. 2)Set up a legal mediation.

What would you do?
```

**Verdict:** KEEP

---

## 520 · `ffb7a681651ee561` · 744c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I threw someone's food out of the work freezer?

```text
On mobile etc. This fish has been sitting there for weeks. It doesn't smell, but I just checked the dates on it, an it should have been used by 9/28 of last year. It's still in it's packaging from wherever they bought it, but hasn't moved in months (apparently). Now, the thing is, there's an ok amount of room in the freezer still. Sure, we need to stack lunches on top of each other, but it's not like the fish is taking up all of our space. If I'm honest, it's annoying me more than anything. No one's eating it, no one seems to care that it's here (except me), and I don't really think anyone should be eating it at this point, even if it has been frozen this whole time.

Would I be wrong if I threw someone's food out of the work freezer?
```

**Verdict:** KEEP

---

## 521 · `ffc520b88462c6d5` · 657c · ask=title · scrubbed

> **Title** (not part of the prompt): WIBTA if I were honest about why I won't go to my friend's show?

```text
I have a couple friends who are in a band and rarely play near our hometown any more. Last night they played and I didn't go because another friend was home Korea for the weekend but they mentioned the band is playing nearby soon. Here's the thing a former friend is also friends with them a n d I have no desire to see this person. Our friendship ended for who knows why, although I accept the bulk of the blame she has gone out of her way to be cruel when I troed to apologize or talk things out. I have never and would never told anyone to choose between us and I know she has.

Would I be wrong if I were honest about why I won't go to my friend's show?
```

**Verdict:** KEEP

---
