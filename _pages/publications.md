---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

<table style="width:100%; border:0; border-spacing:0; border-collapse:separate; margin-right:auto; margin-left:auto; font-size:18px;">
  {% for post in site.publications reversed %}
  <tr>
    <td style="border: none; padding:0; width:25%; vertical-align:middle; max-width:100px; max-height:100px;">
      <div style="position: relative; display: inline-block; margin:0; padding:0;">
        <img src="/{{post.image}}" 
         alt="project image" 
         style="display:block; width:auto; height:auto; max-width:100%; margin:0; padding:0;" />
        {% if post.topic %}
      <div style="position:absolute; top:0; left:0; 
                  background-color: rgba(240,240,240, 0.8); 
                  color:#4CAF50; border:1px solid #4CAF50; 
                  padding:2px 6px; font-size:12px; border-radius:5px; 
                  font-weight:bold;">
        {{ post.topic }}
      </div>
        {% endif %}
      </div>
    </td>
    <td style="border: none; padding:2.5%; width:75%; vertical-align:middle;">
      <h3 style="font-size:22px; margin-bottom:10px;">{{post.title}}</h3>
      <div style="margin-bottom:10px;">{{post.authors}}</div>
      <div style="margin-bottom:10px;"><em>{{post.venue}}</em>, {{ post.date | date: "%Y" }}</div>
      <div style="margin-bottom:10px;">
        {% if post.paper %}
          <a href="{{post.paper}}" style="background-color:#f0f0f0; color:#4CAF50; border:1px solid #4CAF50; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-top:8px; border-radius:5px;">paper</a>
        {% endif %}
        {% if post.code %}
          <a href="{{post.code}}" style="background-color:#f0f0f0; color:#4CAF50; border:1px solid #4CAF50; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px; border-radius:5px;">code</a>
        {% endif %}
        {% if post.web %}
          <a href="{{post.web}}" style="background-color:#f0f0f0; color:#4CAF50; border:1px solid #4CAF50; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px; border-radius:5px;">website</a>
        {% endif %}
        {% if post.video %}
          <a href="{{post.video}}" style="background-color:#f0f0f0; color:#4CAF50; border:1px solid #4CAF50; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px; border-radius:5px;">video</a>
        {% endif %}
        {% if post.poster %}
          / <a href="{{post.poster}}" style="background-color:#f0f0f0; color:#4CAF50; border:1px solid #4CAF50; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px; border-radius:5px;">poster</a>
        {% endif %}
        {% if post.slides %}
          / <a href="{{post.slides}}" style="background-color:#f0f0f0; color:#4CAF50; border:1px solid #4CAF50; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px; border-radius:5px;">slides</a>
        {% endif %}
        {% if post.dataset %}
          / <a href="{{post.dataset}}" style="background-color:#f0f0f0; color:#4CAF50; border:1px solid #4CAF50; padding:4px 10px; text-align:center; text-decoration:none; display:inline-block; margin-left:4px; margin-top:8px; border-radius:5px;">dataset</a>
        {% endif %}
      </div>
      <div>
        {{ post.excerpt }}
      </div>
    </td>
  </tr>
  {% endfor %}
</table>
